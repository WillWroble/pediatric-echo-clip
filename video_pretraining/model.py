"""Set attention encoder for contrastive video pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAttentionEncoder(nn.Module):
    """16 clip embeddings → 1 video embedding via learned query token."""

    def __init__(self, input_dim=768, hidden_dim=768, n_heads=8, dropout=0.2):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection_head = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x):
        """Input (B, 16, 768) → hidden (B, 512)."""
        x = self.project(x)
        q = self.query.expand(x.shape[0], -1, -1)
        x = torch.cat([q, x], dim=1)
        out, _ = self.attention(x, x, x, need_weights=False)
        return self.norm(out[:, 0, :])

    def forward(self, x):
        """Input (B, 16, 768) → L2-normalized (B, 512) for contrastive loss."""
        h = self.encode(x)
        return F.normalize(self.projection_head(h), dim=-1)
