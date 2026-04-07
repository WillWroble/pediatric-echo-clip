"""Line-level decoder: study embedding prepended as position 0, causal self-attention.

Input sequence: [study_emb, cent_0, cent_1, ..., cent_{n-1}]
Output: predicted 768d vector at each position.
Prediction at position i is trained to match cent_i (next centroid).
"""

import torch
import torch.nn as nn
import numpy as np


class LineDecoder(nn.Module):
    def __init__(self, codebook_path, d_model=768, n_heads=8, ff_dim=1024,
                 max_lines=40, dropout=0.1, n_layers=2):
        super().__init__()
        cb = np.load(codebook_path, allow_pickle=True)
        self.register_buffer(
            "centroids",
            torch.from_numpy(cb["centroids"].astype(np.float32)),
        )
        self.pos_emb = nn.Embedding(max_lines + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        n_centroids = self.centroids.shape[0]
        self.head = nn.Linear(d_model, n_centroids)
        self.max_lines = max_lines

    def forward(self, x, mask=None):
        """
        Args:
            x:    (B, L, 768) — [study_emb, cent_0, ..., cent_{n-1}], padded.
            mask: (B, L) bool — True = padded position.
        Returns:
            (B, L, 768) — predicted vectors at each position.
        """
        B, L, D = x.shape
        x = x + self.pos_emb(torch.arange(L, device=x.device))
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.encoder(x, mask=causal, src_key_padding_mask=mask)
        return self.out_proj(h)

    def snap_to_centroid(self, pred):
        """Snap predicted vectors to nearest centroid.

        Args:
            pred: (K, 768)
        Returns:
            indices: (K,) centroid indices.
            distances: (K,) L2 distances to nearest centroid.
        """
        dist = torch.cdist(pred, self.centroids)
        min_dist, indices = dist.min(dim=-1)
        return indices, min_dist
