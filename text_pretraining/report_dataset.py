"""Datasets and data loading for report encoder training.

Two dataloaders over the same preloaded data:
    VICRegDataset:    subsampled lines + masked demos, two views per study
    StandardDataset:  all lines + all demos, variable-length (needs padding collate)

Trajectory pairs are built from StandardDataset's data at construction time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import h5py


TEXT_FIELDS = [
    "summary", "study_findings", "history",
    "measurements", "cardiac_history", "reason_for_exam",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_age(attrs):
    """Use report's age field first, compute from DOB only as fallback."""
    age_str = attrs.get("age", "")
    if age_str:
        age_str = age_str.strip().lower()
        if age_str.endswith("years"):
            try: return float(age_str.replace("years", ""))
            except ValueError: pass
        if age_str.endswith("m"):
            try: return float(age_str[:-1]) / 12.0
            except ValueError: pass
    # fallback: compute from DOB + study_date
    return parse_age_years(attrs.get("dob", ""), attrs.get("study_date", ""))
def parse_age_years(dob_str, study_date_str):
    try:
        sd = datetime.strptime(study_date_str, "%m%d%y%H%M%S")
        for fmt in ["%B %d, %Y", "%d-%b-%Y"]:
            try:
                return (sd - datetime.strptime(dob_str.strip(), fmt)).days / 365.25
            except ValueError:
                continue
    except Exception:
        pass
    return float("nan")


def parse_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def parse_study_date(s):
    try:
        return datetime.strptime(s, "%m%d%y%H%M%S")
    except Exception:
        return None


def study_date_to_year(sd_str):
    """Convert study_date string to fractional year (e.g. 2015.47)."""
    dt = parse_study_date(sd_str)
    if dt is None:
        return float("nan")
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


# ---------------------------------------------------------------------------
# Preloading
# ---------------------------------------------------------------------------

def load_study(grp):
    """Load one HDF5 group → (lines, demos, mrn, study_date) or None."""
    arrays = [grp[f][:] for f in TEXT_FIELDS if f in grp]
    if not arrays:
        return None
    lines = np.concatenate(arrays, axis=0).astype(np.float32)

    attrs = dict(grp.attrs)
    demos = np.array([
        parse_age(attrs),
        1.0 if attrs.get("gender", "").lower() == "male" else (
            0.0 if attrs.get("gender", "").lower() == "female" else float("nan")),
        parse_float(attrs.get("weight_kg", "")),
        parse_float(attrs.get("height_cm", "")),
        parse_float(attrs.get("bsa", "")),
        parse_float(attrs.get("bmi", "")),
    ], dtype=np.float32)

    return lines, demos, attrs.get("mrn", ""), attrs.get("study_date", "")

def load_meta_only(h5_dir, verbose=True):
    """Load only mrn + study_date from HDF5 attrs. Much faster than preload_all."""
    files = sorted(Path(h5_dir).glob("chunk_*.h5"))
    meta = {}
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                attrs = dict(f[sid].attrs)
                meta[sid] = (attrs.get("mrn", ""), attrs.get("study_date", ""))
    if verbose:
        print(f"Loaded metadata for {len(meta):,} studies from {len(files)} chunks", flush=True)
    return meta
def preload_all(h5_dir, min_lines=30, verbose=True):
    """Load all HDF5 chunks into memory.

    Returns:
        data: {study_id: (lines, demos, mrn, study_date)}
    """
    files = sorted(Path(h5_dir).glob("chunk_*.h5"))
    data = {}
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                result = load_study(f[sid])
                if result is not None and result[0].shape[0] >= min_lines:
                    data[sid] = result
    if verbose:
        print(f"Loaded {len(data):,} studies from {len(files)} chunks", flush=True)
    return data


def compute_demo_stats(data):
    """Compute mean/std for demographics, ignoring NaN."""
    all_demos = np.stack([d[1] for d in data.values()])
    mean = np.nanmean(all_demos, axis=0).astype(np.float32)
    std = np.nanstd(all_demos, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_demos(demos, mean, std):
    """Normalize and replace NaN with 0."""
    return np.nan_to_num((demos - mean) / std, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory pair construction
# ---------------------------------------------------------------------------

def build_trajectory_pairs(study_ids, data):
    """Build (sid_t, sid_t1) consecutive pairs from preloaded data.

    Uses mrn + study_date already in the preloaded dict — no HDF5 re-read.
    """
    sid_set = set(study_ids)
    patient_studies = defaultdict(list)
    for sid in sid_set:
        _, _, mrn, sd_str = data[sid]
        dt = parse_study_date(sd_str)
        if mrn and dt:
            patient_studies[mrn].append((dt, sid))

    pairs = []
    for mrn, studies in patient_studies.items():
        studies.sort()
        for i in range(len(studies) - 1):
            pairs.append((studies[i][1], studies[i + 1][1]))
    return pairs

# ---------------------------------------------------------------------------
# Video embedding loading
# ---------------------------------------------------------------------------

def load_video_embeddings(npz_path):
    """Load frozen study-level video embeddings.

    Returns:
        {study_id: (768,) float32 array}
    """
    data = np.load(npz_path)
    embs = data["embeddings"]
    sids = data["study_ids"].astype(str)
    return {sid: emb.astype(np.float32) for sid, emb in zip(sids, embs)}


def load_video_embeddings_by_study(npz_path):
    """Load video-level embeddings grouped by study.

    Returns:
        {study_id: (N_videos, D) float32 array}
    """
    data = np.load(npz_path)
    embs = data["embeddings"]
    sids = data["study_ids"].astype(str)
    by_study = {}
    for emb, sid in zip(embs, sids):
        by_study.setdefault(sid, []).append(emb)
    return {k: np.stack(v, dtype=np.float32) for k, v in by_study.items()}
# ---------------------------------------------------------------------------
# VICReg dataset (fixed-size, augmented)
# ---------------------------------------------------------------------------

class VICRegDataset(Dataset):
    """Two augmented views of the same study: subsampled lines + masked demos."""

    def __init__(self, study_ids, data, demo_mean, demo_std,
                 n_sample=12, demo_dropout=0.3, bernoulli_p=None):
        self.study_ids = study_ids
        self.data = data
        self.demo_mean = demo_mean
        self.demo_std = demo_std
        self.n_sample = n_sample
        self.demo_dropout = demo_dropout
        self.bernoulli_p = bernoulli_p

    def __len__(self):
        return len(self.study_ids)

    def _sample_lines(self, lines):
        if self.bernoulli_p is not None:
            mask = np.random.rand(lines.shape[0]) < self.bernoulli_p
            if mask.sum() == 0:
                mask[np.random.randint(lines.shape[0])] = True
            return lines[mask]
        n = lines.shape[0]
        sel = np.random.choice(n, self.n_sample, replace=n < self.n_sample)
        return lines[sel]

    def _mask_demos(self, demos_t):
        mask = (torch.rand(demos_t.shape[0]) > self.demo_dropout).float()
        return demos_t * mask

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        lines, demos_raw, _, sd_str = self.data[sid]
        demos = torch.from_numpy(normalize_demos(demos_raw, self.demo_mean, self.demo_std))

        lines1 = torch.from_numpy(self._sample_lines(lines))
        lines2 = torch.from_numpy(self._sample_lines(lines))
        demos1 = self._mask_demos(demos)
        demos2 = self._mask_demos(demos)

        t = torch.tensor(study_date_to_year(sd_str), dtype=torch.float32)
        return lines1, demos1, lines2, demos2, t

def vicreg_collate(batch):
    """Pad variable-length VICReg views and build masks.

    Returns:
        lines1, demos1, mask1, lines2, demos2, mask2, t
    """
    lines1_list, demos1_list, lines2_list, demos2_list, t_list = zip(*batch)

    def pad_and_mask(lines_list):
        lengths = [l.shape[0] for l in lines_list]
        padded = pad_sequence(lines_list, batch_first=True, padding_value=0.0)
        mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
        for i, ln in enumerate(lengths):
            mask[i, ln:] = True
        return padded, mask

    lines1, mask1 = pad_and_mask(lines1_list)
    lines2, mask2 = pad_and_mask(lines2_list)
    demos1 = torch.stack(demos1_list)
    demos2 = torch.stack(demos2_list)
    t = torch.stack(t_list)

    return lines1, demos1, mask1, lines2, demos2, mask2, t
# ---------------------------------------------------------------------------
# Standard dataset (variable-length, no augmentation)
# ---------------------------------------------------------------------------

class StandardDataset(Dataset):
    """All lines + all demos per study. Variable-length, requires padded collate."""

    def __init__(self, study_ids, data, demo_mean, demo_std):
        self.study_ids = study_ids
        self.data = data
        self.demo_mean = demo_mean
        self.demo_std = demo_std

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        lines, demos_raw, _, _ = self.data[sid]
        lines_t = torch.from_numpy(lines)
        demos_t = torch.from_numpy(normalize_demos(demos_raw, self.demo_mean, self.demo_std))
        return lines_t, demos_t, sid


def standard_collate(batch):
    """Pad variable-length lines and build attention mask.

    Returns:
        lines:  (B, max_N, 768) padded
        demos:  (B, 6)
        mask:   (B, max_N) bool, True = padding
        sids:   list[str]
    """
    lines_list, demos_list, sids = zip(*batch)
    lengths = [l.shape[0] for l in lines_list]
    lines = pad_sequence(lines_list, batch_first=True, padding_value=0.0)
    demos = torch.stack(demos_list)

    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for i, ln in enumerate(lengths):
        mask[i, ln:] = True

    return lines, demos, mask, list(sids)


# ---------------------------------------------------------------------------
# Trajectory dataset (pairs from standard data)
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Consecutive report pairs. Variable-length, requires padded collate."""

    def __init__(self, pairs, data, demo_mean, demo_std):
        self.pairs = pairs
        self.data = data
        self.demo_mean = demo_mean
        self.demo_std = demo_std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sid_t, sid_t1 = self.pairs[idx]
        lines_t, demos_t, _, _ = self.data[sid_t]
        lines_t1, demos_t1, _, _ = self.data[sid_t1]
        return (
            torch.from_numpy(lines_t),
            torch.from_numpy(normalize_demos(demos_t, self.demo_mean, self.demo_std)),
            torch.from_numpy(lines_t1),
            torch.from_numpy(normalize_demos(demos_t1, self.demo_mean, self.demo_std)),
        )


def trajectory_collate(batch):
    """Pad both studies in each pair.

    Returns:
        lines_t, demos_t, mask_t, lines_t1, demos_t1, mask_t1
    """
    lines_t_list, demos_t_list, lines_t1_list, demos_t1_list = zip(*batch)

    def pad_and_mask(lines_list):
        lengths = [l.shape[0] for l in lines_list]
        padded = pad_sequence(lines_list, batch_first=True, padding_value=0.0)
        mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
        for i, ln in enumerate(lengths):
            mask[i, ln:] = True
        return padded, mask

    lines_t, mask_t = pad_and_mask(lines_t_list)
    lines_t1, mask_t1 = pad_and_mask(lines_t1_list)
    demos_t = torch.stack(demos_t_list)
    demos_t1 = torch.stack(demos_t1_list)

    return lines_t, demos_t, mask_t, lines_t1, demos_t1, mask_t1

# ---------------------------------------------------------------------------
# Contrastive dataset (report ↔ video alignment)
# ---------------------------------------------------------------------------

class ContrastDataset(Dataset):
    """Report + frozen video embedding pairs. All lines, padded via collate."""

    def __init__(self, study_ids, data, video_embs, demo_mean, demo_std):
        self.study_ids = [s for s in study_ids if s in video_embs]
        self.data = data
        self.video_embs = video_embs
        self.demo_mean = demo_mean
        self.demo_std = demo_std

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        lines, demos_raw, _, _ = self.data[sid]
        demos = normalize_demos(demos_raw, self.demo_mean, self.demo_std)
        video = self.video_embs[sid]
        return (
            torch.from_numpy(lines),
            torch.from_numpy(demos),
            torch.from_numpy(video),
        )


def contrast_collate(batch):
    """Pad report lines, stack video targets.

    Returns:
        lines: (B, max_N, 768) padded
        demos: (B, 6)
        mask:  (B, max_N) bool, True = padding
        video: (B, 768) frozen target
    """
    lines_list, demos_list, video_list = zip(*batch)
    lengths = [l.shape[0] for l in lines_list]
    lines = pad_sequence(lines_list, batch_first=True, padding_value=0.0)
    demos = torch.stack(demos_list)
    video = torch.stack(video_list)

    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for i, ln in enumerate(lengths):
        mask[i, ln:] = True

    return lines, demos, mask, video

class VideoContrastDataset(Dataset):
    """Report + video-level embeddings for joint contrastive training."""

    def __init__(self, study_ids, data, video_embs, demo_mean, demo_std,
                 n_sample=12, n_videos_sample=12):
        self.study_ids = [s for s in study_ids if s in video_embs]
        self.data = data
        self.video_embs = video_embs
        self.demo_mean = demo_mean
        self.demo_std = demo_std
        self.n_sample = n_sample
        self.n_videos_sample = n_videos_sample

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        lines, demos_raw, _, _ = self.data[sid]
        demos = normalize_demos(demos_raw, self.demo_mean, self.demo_std)
        # sample videos
        videos = self.video_embs[sid]
        nv = videos.shape[0]
        vsel = np.random.choice(nv, self.n_videos_sample, replace=nv < self.n_videos_sample)

        return (
            torch.from_numpy(lines),
            torch.from_numpy(demos),
            torch.from_numpy(videos[vsel]),
        )

def video_contrast_collate(batch):
    lines_list, demos_list, video_list = zip(*batch)
    lengths = [l.shape[0] for l in lines_list]
    lines = pad_sequence(lines_list, batch_first=True, padding_value=0.0)
    demos = torch.stack(demos_list)
    video = torch.stack(video_list)

    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for i, ln in enumerate(lengths):
        mask[i, ln:] = True

    return lines, demos, mask, video
# report_dataset.py — add after imports

class PatientBatchSampler(torch.utils.data.Sampler):
    """Yield indices such that each batch has one study per unique patient."""

    def __init__(self, study_ids, data, batch_size=512):
        self.batch_size = batch_size
        # group study indices by MRN
        self.mrn_to_idxs = defaultdict(list)
        for i, sid in enumerate(study_ids):
            mrn = data[sid][2]
            if mrn:
                self.mrn_to_idxs[mrn].append(i)
        self.mrns = list(self.mrn_to_idxs.keys())

    def __iter__(self):
        rng = np.random.default_rng()
        rng.shuffle(self.mrns)
        for start in range(0, len(self.mrns), self.batch_size):
            batch = []
            for mrn in self.mrns[start:start + self.batch_size]:
                idxs = self.mrn_to_idxs[mrn]
                batch.append(idxs[rng.integers(len(idxs))])
            yield batch

    def __len__(self):
        return len(self.mrns) // self.batch_size
