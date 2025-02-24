import torch
import torch.nn as nn
import math

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create relative positional encodings
        pe = torch.zeros(2 * max_len - 1, d_model)
        position = torch.arange(-max_len + 1, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # Create relative position indices
        indices = torch.arange(seq_len, dtype=torch.long)
        relative_indices = indices.unsqueeze(0) - indices.unsqueeze(1) + max_len - 1
        relative_indices = relative_indices.to(x.device)
        # Apply relative positional encoding
        x = x + self.pe[relative_indices]
        return self.dropout(x)
