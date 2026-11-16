"""model.py: one multilingual encoder, two classification heads (category, sentiment)."""
import torch
from transformers import AutoModel


class TicketClassifier(torch.nn.Module):
    def __init__(self, base, n_category, n_sentiment, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.drop = torch.nn.Dropout(dropout)
        self.category = torch.nn.Linear(h, n_category)
        self.sentiment = torch.nn.Linear(h, n_sentiment)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        cls = self.drop(cls)
        return self.category(cls), self.sentiment(cls)
