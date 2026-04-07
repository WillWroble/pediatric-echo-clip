"""EchoFocus Mini — transformer over shaped video embeddings."""

import torch.nn as nn
import torch.nn.functional as F


class EchoFocus(nn.Module):
    def __init__(self, input_dim=512, n_heads=8, ff_dim=512, dropout=0.2, n_targets=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, n_targets)
        self.contrast_proj = nn.Linear(input_dim, input_dim)
    def encode(self, x):
        """x: (B, N_videos, D) → (B, D)"""
        h = self.encoder(x)
        h = h.mean(dim=1)
        return self.norm(h)

    def forward(self, x):
        """x: (B, N_videos, D) → (B, n_targets)"""
        return self.head(self.encode(x))

    def contrast(self, x):
        """x: (B, N_videos, D) → (B, D) L2-normalized for InfoNCE."""
        return F.normalize(self.contrast_proj(F.gelu(self.encode(x))), dim=-1)
