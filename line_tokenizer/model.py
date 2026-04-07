"""LineEncoder: frozen ClinicalBERT (last layer unfrozen) → CLS → projection → 768d."""

import torch.nn as nn
from transformers import AutoModel


class LineEncoder(nn.Module):

    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)
