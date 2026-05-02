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

    def attention(query, key, value, mask, dropout:float):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_length, seq_length)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float(-1e9))  # (batch_size, h, seq_length, seq_length)
        attention_score = F.softmax(scores, dim=-1)  # (batch_size, h, seq_length, seq_length)
        if dropout > 0:
            attention_score = F.dropout(attention_score, p=dropout, training=self.training)  # (batch_size, h, seq_length, seq_length)
        out = torch.matmul(attention_score, value)  # (batch_size, h, seq_length, d_k)
        return out, attention_score

    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_length, d_model)
        key = self.w_k(k) # (batch_size, seq_length, d_model)
        value = self.w_v(v) # (batch_size, seq_length, d_model)

        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_length, d_k)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_length, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_length, d_k)
        x, self.attention_score = self.attention(query, key, value, mask, self.dropout.p)  # x (batch_size, h, seq_length, d_k)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(2), self.h * self.d_k)  # (batch_size, seq_length, d_model)
        return self.w_o(x)  # (batch_size, seq_length, d_model)
    
class ResidualConnection(nn.Module):
    def __init__(self, size:int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNormalization(size)

    def forward(self, x, sublayer):
        out = x + self.dropout(sublayer(x))
        return self.ln(out)

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.add_norm_attn = ResidualConnection(size=d_model, dropout=dropout)
        self.add_norm_ffn = ResidualConnection(size=d_model, dropout=dropout)
    def forward(self, x, mask):
        x = self.add_norm_attn(x, lambda x: self.attention(x, x, x, mask))  # (batch_size, seq_length, d_model)
        x = self.add_norm_ffn(x, self.feed_forward)  # (batch_size, seq_length, d_model)
        return x

class Encoder(nn.Module):
    def __init__(self,d_model:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self,x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, self_attetion:MultiHeadAttention, cross_attention:MultiHeadAttention, ffn:FeedForward, dropout:float):
        super().__init__()
        self.attention = self_attetion
        self.cross_attention = cross_attention
        self.feed_forward = ffn
        self.add_norm_cross_MHA = ResidualConnection(d_model, dropout)
        self.add_norm_masked_MHA = ResidualConnection(d_model, dropout)
        self.add_norm_feed_forward = ResidualConnection(d_model, dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.add_norm_masked_MHA(x, lambda x: self.attention(x, x, x, tgt_mask))
        x = self.add_norm_cross_MHA(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.add_norm_feed_forward(x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, encoder_ouput, src_mask, tgt_mask):
        for layer in  self.layers:
            x = layer(x, encoder_ouput, src_mask, tgt_mask )
        return self.norm(x)
    
class Projection(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_positional_encoding: PositionalEncoding, tgt_positional_encoding: PositionalEncoding, projection: Projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection = projection
    
    def encode(self, src, src_mask):
        embedded_src = self.src_embedding(src)  # (batch_size, seq_length, d_model)
        encoded_src = self.src_positional_encoding(embedded_src)  # (batch_size, seq_length, d_model)
        return self.encoder(encoded_src, src_mask)  # (batch_size, seq_length, d_model)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        embedded_tgt = self.tgt_embedding(tgt)  # (batch_size, seq_length, d_model)
        decoded_tgt = self.tgt_positional_encoding(embedded_tgt)  # (batch_size, seq_length, d_model)
        return self.decoder(decoded_tgt, encoder_output, src_mask, tgt_mask)  # (batch_size, seq_length, d_model)
    def project(self, decoder_output):
        return self.projection(decoder_output)  # (batch_size, seq_length, vocab_size)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):

    src_embedding = InputEmbedding(d_model,src_vocab_size)
    tgt_embedding = InputEmbedding(d_model,tgt_vocab_size)

    src_pos_enconding = PositionalEncoding(d_model, src_seq_len)
    tgt_pos_enconding = PositionalEncoding(d_model, tgt_seq_len)

    encoder_blocks= []
    for _ in range(N):
        encoder_attention = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, encoder_attention, encoder_feed_forward, dropout))

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    
    decoder_blocks= []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(d_model, decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout))

    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection = Projection(d_model, tgt_vocab_size)

    Transformer_model = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos_enconding, tgt_pos_enconding, projection)

    for p in Transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return Transformer_model