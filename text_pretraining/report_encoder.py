"""Report encoder: Set Transformer over ClinicalBERT line embeddings.

Architecture:
    lines → MHA(query + lines) → LayerNorm(CLS) → cat(demos) → fuse → 768d
                                                                  ├→ vicreg_proj
                                                                  └→ traj_proj (L2-normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReportEncoder(nn.Module):
    """Encode variable-length line embeddings + demographics → fixed 768d vector.

    Three output paths from the same 768d fused representation:
        encode():  returns 768d as-is (downstream / analysis)
        vicreg():  linear projection (trained with VICReg loss)
        traj():    linear projection + L2 norm (trained with cosine delta loss)
    """

    def __init__(self, input_dim=768, n_heads=8, dropout=0.2, n_demo=6, output_dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            input_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.fuse = nn.Sequential(
            nn.Linear(input_dim + n_demo, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.vicreg_proj = nn.Linear(output_dim, output_dim)
        self.traj_proj = nn.Linear(output_dim, output_dim)
        self.contrast_proj = nn.Linear(output_dim, output_dim)

    def encode(self, lines, demos, mask=None):
        """(B, N, 768) + (B, 6) → (B, 768). No projection, just the fused embedding.

        Args:
            lines: (B, N, 768) ClinicalBERT line embeddings.
            demos: (B, 6) normalized demographics.
            mask:  (B, N) bool, True = padded (ignored) position.
        """
        B = lines.shape[0]
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, lines], dim=1)  # (B, 1+N, D)

        # Build key_padding_mask: (B, 1+N), False=attend, True=ignore
        if mask is not None:
            q_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([q_mask, mask], dim=1)
        else:
            key_padding_mask = None

        out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = self.norm(out[:, 0, :])
        return self.fuse(torch.cat([pooled, demos], dim=-1))

    def vicreg(self, lines, demos, mask=None):
        """encode → vicreg projection."""
        return self.vicreg_proj(F.gelu(self.encode(lines, demos, mask)))

    def traj(self, lines, demos, mask=None):
        """encode → trajectory projection."""
        return self.traj_proj(F.gelu(self.encode(lines, demos, mask)))
    def contrast(self, lines, demos, mask=None):
        """encode → contrast projection + L2 norm for InfoNCE."""
        return F.normalize(self.contrast_proj(F.gelu(self.encode(lines, demos, mask))), dim=-1)
