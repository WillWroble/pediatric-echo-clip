from transformers import AutoModel
import torch

bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
torch.save(bert.embeddings.word_embeddings.state_dict(), "clinicalbert_word_embeddings.pt")
print(f"Saved: {bert.embeddings.word_embeddings.weight.shape}")
del bert
