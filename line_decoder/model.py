"""LineDecoder: study embedding → 10K clinical finding scores.

Architecture: Linear(768→768) → GELU → Linear(768→vocab_size)
Equivalent to vocab_size simultaneous GELU probes with shared hidden layer.
"""

import torch
import torch.nn as nn


class LineDecoder(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=768, vocab_size=10000, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        """(B, 768) → (B, vocab_size) raw logits."""
        return self.net(x)
