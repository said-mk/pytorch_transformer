import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class InputEmbedding(nn.Module):
    """Embedding layer with scaling for input tokens."""
    def __init__(self, d_model: int, vocab_size:int):
        """
        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # (vocab_size, d_model)
    
    def forward(self, x):
        """Forward pass for input embedding.
        Args:
            x (Tensor): Input tensor of token indices. Shape: (batch_size, seq_length)
        Returns:
            Tensor: Embedded input. Shape: (batch_size, seq_length, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_length, d_model)

class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input tensor."""
    def __init__(self, d_model:int, seq_length:int, dropout:float):
        """
        Args:
            d_model (int): Dimension of the model.
            seq_length (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_length, d_model)  # (seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)  # (seq_length, 1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))  # (d_model/2,)
        
        pe[:, 0::2] = torch.sin(position * divisor)  # (seq_length, d_model/2)
        pe[:, 1::2] = torch.cos(position * divisor)  # (seq_length, d_model/2)

        pe = pe.unsqueeze(0)  # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass for positional encoding.
        Args:
            x (Tensor): Input tensor. Shape: (batch_size, seq_length, d_model)
        Returns:
            Tensor: Output tensor with positional encoding added. Shape: (batch_size, seq_length, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :])  # (batch_size, seq_length, d_model)
        return self.dropout(x)  # (batch_size, seq_length, d_model)

class LayerNormalization(nn.Module):
    """Applies layer normalization over a batch of inputs."""
    def __init__(self, features: int, eps: float = 1e-6):
        """
        Args:
            features (int): Number of features(d_model) in the input tensor.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(features))  # (features,)
        self.gamma = nn.Parameter(torch.ones(features))  # (features,)

    def forward(self, x):
        """Forward pass for layer normalization.
        Args:
            x (Tensor): Input tensor. Shape: (..., features)
        Returns:
            Tensor: Layer-normalized tensor. Shape: (..., features)
        """
        x_mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        x_std = x.std(dim=-1, unbiased=False, keepdim=True)  # (..., 1)
        # Normalize and scale
        return self.gamma * ((x - x_mean) / (x_std + self.eps)) + self.beta  # (..., features)

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(d_model, self.d_model, bias=False)

    @staticmethod
    def attention(query, key, value, mask, dropout:float):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_length, seq_length)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # (batch_size, h, seq_length, seq_length)
        attention_score = F.softmax(scores, dim=-1)  # (batch_size, h, seq_length, seq_length)
        if dropout > 0:
            attention_score = F.dropout(attention_score, p=dropout)  # (batch_size, h, seq_length, seq_length)
        output = torch.matmul(attention_score, value)  # (batch_size, h, seq_length, d_k)
        return output, attention_score

    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_length, d_model)
        key = self.w_k(k) # (batch_size, seq_length, d_model)
        value = self.w_v(v) # (batch_size, seq_length, d_model)

        query = query.view(-1, query.size(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_length, d_k)
        key = key.view(-1, key.size(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_length, d_k)
        value = value.view(-1, value.size(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_length, d_k)
        x, self.attention_score = self.attention(query, key, value, mask, self.dropout.p)  # x (batch_size, h, seq_length, d_k)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(2), self.h * self.d_k)  # (batch_size, seq_length, d_model)
        return self.w_o(x)  # (batch_size, seq_length, d_model)
    