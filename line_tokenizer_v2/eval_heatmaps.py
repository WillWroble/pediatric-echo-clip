"""
AUROC analysis for EchoVQ heatmap pathology detection.

Unit of analysis: (study, concept) pair
  score = max hotspot-line score matching the concept (across all fields)
  label = concept positively present in reference (not ruled-out / negated / normal)

Two labeling scopes reported:
  echo-only  – reference from findings + summary
  all-fields – reference from findings + summary + history
"""
import json, re
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

FIELDS = ["study_findings", "summary", "history"]
ECHO_FIELDS = ["study_findings", "summary"]

CONCEPTS = [
    # valve regurgitation
    ("tricuspid_regurg",     r"tricuspid\s+regurgitation"),
    ("mitral_regurg",        r"mitral\s+regurgitation"),
    ("aortic_regurg",        r"aortic\s+regurgitation"),
    ("pulmonary_regurg",     r"pulmonary\s+regurgitation"),
    # valve stenosis
    ("aortic_stenosis",      r"aortic\s+(?:valve\s+)?stenosis"),
    ("pulmonary_stenosis",   r"pulmonary\s+(?:valve\s+)?stenosis"),
    ("mitral_stenosis",      r"mitral\s+stenosis"),
    # shunts / defects
    ("pda",                  r"patent\s+ductus\s+arteriosus"),
    ("pfo",                  r"patent\s+foramen\s+ovale"),
    ("asd",                  r"atrial\s+septal\s+defect"),
    ("vsd",                  r"ventricular\s+septal\s+defect"),
    # chamber dilation
    ("dilated_lv",           r"dilated\s+left\s+ventricle|left\s+ventricul\w+\s+dilat"),
    ("dilated_rv",           r"dilated\s+right\s+ventricle|right\s+ventricul\w+\s+dilat"),
    ("dilated_la",           r"dilated\s+left\s+atrium|left\s+atrial\s+dilat"),
    ("dilated_ra",           r"dilated\s+right\s+atrium|right\s+atrial\s+dilat"),
    ("dilated_ao_root",      r"dilated\s+aortic\s+root"),
    ("dilated_asc_ao",       r"dilated\s+ascending\s+aorta"),
    # dysfunction
    ("lv_sys_dysfunction",   r"left\s+ventricul\w+\s+(?:\w+\s+)?(?:systolic\s+)?dysfunction"),
    ("rv_sys_dysfunction",   r"right\s+ventricul\w+\s+(?:\w+\s+)?(?:systolic\s+)?dysfunction"),
    ("lv_diast_dysfunction", r"diastolic\s+(?:left\s+ventricul\w+\s+)?dysfunction"
                             r"|left\s+ventricul\w+\s+diastolic\s+dysfunction"),
    # pressure / hypertrophy
    ("pulm_hypertension",    r"(?:pulmonary|right\s+ventricul\w+)\s+hypertension"),
    ("rv_hypertrophy",       r"right\s+ventricul\w+\s+hypertrophy"),
    ("lv_hypertrophy",       r"left\s+ventricul\w+\s+hypertrophy"),
    # structural
    ("coarctation",          r"coarctation"),
    ("pericardial_effusion", r"pericardial\s+effusion"),
    ("lvoto",                r"left\s+ventricular\s+outflow\s+tract\s+obstruction"),
]

NEG_RE = re.compile(
    r"ruled\s+out"
    r"|(?:^|[•\s])no\s+"
    r"|no\s+evidence"
    r"|no\s+significant"
    r"|no\s+echocardiographic\s+evidence"
    r"|cannot\s+be\s+excluded"
    r"|unable\s+to\s+exclude",
    re.IGNORECASE,
)
NORMAL_RE = re.compile(r",\s*normal\b|function,?\s*normal", re.IGNORECASE)


def is_positive(text, cre):
    t = text.lower()
    return bool(cre.search(t)) and not NEG_RE.search(t) and not NORMAL_RE.search(t)


def compute(data, compiled, label_fields, score_fields):
    per = defaultdict(lambda: {"s": [], "l": []})
    for study in data:
        for name, cre in compiled:
            # label: positive in any label-field reference?
            lbl = 0
            for f in label_fields:
                if any(is_positive(r["text"], cre) for r in study[f]["reference"]):
                    lbl = 1
                    break
            # score: max hotspot-line score in any score-field
            best = 0.0
            for f in score_fields:
                for hs in study[f]["hotspots"]:
                    for line in hs:
                        if cre.search(line["text"].lower()):
                            best = max(best, line["score"])
            per[name]["s"].append(best)
            per[name]["l"].append(lbl)
    return per


def report(per, compiled, title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"{'concept':<25} {'AUROC':>7} {'p@.9':>6} {'r@.9':>6}"
          f" {'n_pos':>6} {'n_neg':>6} {'prev':>6}")
    print("-" * 72)

    rows = []
    for name, _ in compiled:
        s, l = np.array(per[name]["s"]), np.array(per[name]["l"])
        n_pos = int(l.sum())
        n = len(l)
        auc = roc_auc_score(l, s) if 0 < n_pos < n else None
        m = s >= 0.9
        prec = float(l[m].mean()) if m.sum() and l[m].sum() else 0.0
        rec = float(l[m].sum() / l.sum()) if l.sum() else 0.0
        rows.append((name, auc, prec, rec, n_pos, n - n_pos, n_pos / n))

    rows.sort(key=lambda r: r[1] if r[1] else -1, reverse=True)
    aucs = []
    for name, auc, prec, rec, n_pos, n_neg, prev in rows:
        a = f"{auc:.3f}" if auc else " N/A"
        print(f"{name:<25} {a:>7} {prec:>6.3f} {rec:>6.3f}"
              f" {n_pos:>6} {n_neg:>6} {prev:>6.3f}")
        if auc:
            aucs.append(auc)

    print("-" * 72)
    # macro (all)
    print(f"{'macro (all)':<25} {np.mean(aucs):.3f}")
    # macro (prev < 50%)
    lp = [a for (_, a, _, _, _, _, p) in rows if a and p < 0.5]
    if lp:
        print(f"{'macro (prev < 50%)':<25} {np.mean(lp):.3f}   ({len(lp)} concepts)")
    # micro
    all_s = np.concatenate([np.array(per[n]["s"]) for n, _ in compiled])
    all_l = np.concatenate([np.array(per[n]["l"]) for n, _ in compiled])
    print(f"{'micro (pooled)':<25} {roc_auc_score(all_l, all_s):.3f}"
          f"   ({len(all_s)} pairs, prev={all_l.mean():.3f})")
    return aucs


def main():
    with open("/mnt/user-data/uploads/line_heatmaps_merged.json") as f:
        data = json.load(f)
    print(f"Studies: {len(data)}  |  Concepts: {len(CONCEPTS)}")

    compiled = [(n, re.compile(p, re.IGNORECASE)) for n, p in CONCEPTS]

    per_echo = compute(data, compiled, ECHO_FIELDS, FIELDS)
    report(per_echo, compiled, "ECHO-ONLY LABELS  (findings + summary ref, all-field hotspots)")

    per_all = compute(data, compiled, FIELDS, FIELDS)
    report(per_all, compiled, "ALL-FIELDS LABELS  (all ref, all-field hotspots)")


if __name__ == "__main__":
    main()
