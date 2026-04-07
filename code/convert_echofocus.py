"""Convert EchoFocus HDF5 to .npz matching PanEcho format."""

import numpy as np
import h5py
from pathlib import Path

IN_PATH = Path("/lab-share/Cardio-Mayourian-e2/Public/EchoFocus_Features/Bill_Features/Bill_Feature_Out.hdf5")
OUT_PATH = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/echofocus_embeddings.npz")

with h5py.File(IN_PATH, "r") as f:
    study_ids = f["EID"][:].astype(str)
    embeddings = f["Features"][:].astype(np.float32)

print(f"Converted: {embeddings.shape[0]} studies, {embeddings.shape[1]}-dim")
np.savez(OUT_PATH, study_ids=study_ids, embeddings=embeddings)
print(f"Saved to {OUT_PATH}")
