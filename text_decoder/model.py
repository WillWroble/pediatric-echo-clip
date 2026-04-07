import torch
import torch.nn as nn


class ReportDecoder(nn.Module):
    def __init__(self, d_model=768, n_heads=8, ff_dim=1024, n_layers=2,
         max_seq_len=512, dropout=0.1, emb_path="clinicalbert_word_embeddings.pt"):
        
        super().__init__()

        # frozen ClinicalBERT token embeddings (tied to output)

        self.token_emb = nn.Embedding(28996, d_model)
        self.token_emb.load_state_dict(torch.load(emb_path, weights_only=True))
        self.token_emb.requires_grad_(False)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, cond, pad_mask=None):
        """
        tokens:   (B, T) long — input token IDs
        cond:     (B, S, D) float — conditioning signal (S=1 for study emb)
        pad_mask: (B, T) bool — True at pad positions
        Returns:  (B, T, vocab_size) logits
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.emb_drop(self.token_emb(tokens) + self.pos_emb(pos))
        x = torch.cat([cond, x], dim=1)  # (B, S+T, D)
        S = cond.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(S + T, device=tokens.device)
        if pad_mask is not None:
            prefix = torch.zeros(B, S, dtype=torch.bool, device=tokens.device)
            pad_mask = torch.cat([prefix, pad_mask], dim=1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        x = self.norm(x[:, S:, :])  # slice off conditioning prefix
        return x @ self.token_emb.weight.T

