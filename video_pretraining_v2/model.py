"""ClipAggregator: mirrors JEPA's AttentiveClassifier but replaces the linear head
with LayerNorm + projection + L2-norm for InfoNCE training."""

import torch.nn as nn
import torch.nn.functional as F

from src.models.attentive_pooler import AttentivePooler


class ClipAggregator(nn.Module):
    def __init__(self, embed_dim=768, num_heads=16, mlp_ratio=4.0, depth=4,
                 complete_block=True):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            complete_block=complete_block,
            use_activation_checkpointing=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.projection_head = nn.Linear(embed_dim, embed_dim)

    def encode(self, x):
        """Pre-projection embedding (used at extraction time)."""
        return self.norm(self.pooler(x).squeeze(1))

    def forward(self, x):
        """L2-normalized post-projection embedding (used for InfoNCE loss)."""
        return F.normalize(self.projection_head(self.encode(x)), dim=-1)
