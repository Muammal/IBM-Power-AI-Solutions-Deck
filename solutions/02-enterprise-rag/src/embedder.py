"""embedder.py: sentence embeddings with plain transformers (no sentence-transformers dependency).

Granite Embedding 278M multilingual is an XLM-RoBERTa encoder; its sentence vector is the [CLS] token,
L2-normalised, so cosine similarity = dot product. On Power10 the encoder's matrix multiplies run through
OpenBLAS POWER10 kernels (MMA) inside IBM's torch wheel.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Embedder:
    def __init__(self, model_id, pooling="cls", max_length=512, batch_size=32):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.pooling, self.max_length, self.batch_size = pooling, max_length, batch_size

    def pool(self, out, mask):
        if self.pooling == "cls":
            v = out.last_hidden_state[:, 0]
        else:  # mean over real tokens
            m = mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            v = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(v, dim=-1)

    def forward(self, texts):
        """Differentiable embeddings (used by fine-tuning)."""
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return self.pool(self.model(**enc), enc["attention_mask"])

    @torch.inference_mode()
    def encode(self, texts):
        vecs = [self.forward(texts[i:i + self.batch_size]).float().numpy()
                for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(vecs) if vecs else np.zeros((0, self.model.config.hidden_size), np.float32)
