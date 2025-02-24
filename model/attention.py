import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.window_size = window_size

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None, dropout=None):
        bs = q.size(0)

        # Perform linear operation and split into num_heads
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k)

        # Transpose to get dimensions bs * num_heads * seq_len * d_k
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply sliding window attention
        attended_values = []
        for i in range(0, q.size(2), self.window_size):
            q_window = q[:, :, i:i+self.window_size, :]
            k_window = k[:, :, i:i+self.window_size, :]
            v_window = v[:, :, i:i+self.window_size, :]
            scores = self.attention(q_window, k_window, v_window, self.d_k, mask, dropout)
            attended_values.append(scores)

        # Concatenate attended values
        concat = torch.cat(attended_values, dim=2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        output = self.out(concat)
        return output
