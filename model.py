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
        x_mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        x_std = x.std(dim=-1, unbiased=False, keepdim=True)  # (..., 1)
        # Normalize and scale
        return self.gamma * ((x - x_mean) / (x_std + self.eps)) + self.beta  # (..., features)

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # (d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)  # (d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x) # (batch_size, seq_length, d_ff)
        x = F.relu(x) # (batch_size, seq_length, d_ff)
        x = self.dropout(x) # (batch_size, seq_length, d_ff)
        x = self.linear2(x) # (batch_size, seq_length, d_model)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self/cross attention mechanism."""
    def __init__(self, d_model:int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, self.d_model, bias=False)  # (d_model, d_model)
        self.w_k = nn.Linear(d_model, self.d_model, bias=False)  # (d_model, d_model)
        self.w_v = nn.Linear(d_model, self.d_model, bias=False)  # (d_model, d_model)
        self.w_o = nn.Linear(d_model, self.d_model, bias=False)  # (d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout:float):
        """
        Scaled dot-product attention.
        Args:
            query (Tensor): (batch_size, h, seq_length, d_k)
            key (Tensor): (batch_size, h, seq_length, d_k)
            value (Tensor): (batch_size, h, seq_length, d_k)
            mask (Tensor or None): (batch_size, 1, seq_length, seq_length)
            dropout (float): Dropout rate.
        Returns:
            out (Tensor): (batch_size, h, seq_length, d_k)
            attention_score (Tensor): (batch_size, h, seq_length, seq_length)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_length, seq_length)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float(-1e9))  # (batch_size, h, seq_length, seq_length)
        attention_score = F.softmax(scores, dim=-1)  # (batch_size, h, seq_length, seq_length)
        if dropout > 0:
            attention_score = F.dropout(attention_score, p=dropout, training=True)  # (batch_size, h, seq_length, seq_length)
        out = torch.matmul(attention_score, value)  # (batch_size, h, seq_length, d_k)
        return out, attention_score

    def forward(self, q, k, v, mask):
        """
        Args:
            q, k, v (Tensor): (batch_size, seq_length, d_model)
            mask (Tensor or None): (batch_size, 1, seq_length, seq_length)
        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
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
    """Residual connection followed by layer normalization."""
    def __init__(self, size:int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNormalization(size)

    def forward(self, x, sublayer):
        out = x + self.dropout(sublayer(x))
        return self.ln(out) # (batch_size, seq_length, d_model)

class EncoderBlock(nn.Module):
    """Single encoder block: self-attention + feed-forward."""
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
    """Transformer encoder: stack of encoder blocks."""
    def __init__(self,d_model:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self,x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # (batch_size, seq_length, d_model)

class DecoderBlock(nn.Module):
    """Single decoder block: masked self-attention, cross-attention, feed-forward."""
    def __init__(self, d_model:int, self_attetion:MultiHeadAttention, cross_attention:MultiHeadAttention, ffn:FeedForward, dropout:float):
        
        super().__init__()
        self.attention = self_attetion
        self.cross_attention = cross_attention
        self.feed_forward = ffn
        self.add_norm_cross_MHA = ResidualConnection(d_model, dropout)
        self.add_norm_masked_MHA = ResidualConnection(d_model, dropout)
        self.add_norm_feed_forward = ResidualConnection(d_model, dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x (Tensor): (batch_size, seq_length, d_model)
            encoder_output (Tensor): (batch_size, seq_length, d_model)
            src_mask (Tensor): (batch_size, 1, seq_length, seq_length)
            tgt_mask (Tensor): (batch_size, 1, seq_length, seq_length)
        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
        x = self.add_norm_masked_MHA(x, lambda x: self.attention(x, x, x, tgt_mask))
        x = self.add_norm_cross_MHA(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.add_norm_feed_forward(x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """Transformer decoder: stack of decoder blocks."""
    def __init__(self, d_model:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, encoder_ouput, src_mask, tgt_mask):
        for layer in  self.layers:
            x = layer(x, encoder_ouput, src_mask, tgt_mask )
        return self.norm(x) # (batch_size, seq_length, d_model)

class Projection(nn.Module):
    """Linear projection to vocabulary size with log-softmax."""
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)  # (d_model, vocab_size)
    
    def forward(self, x):
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1) #(batch_size, seq_length, vocab_size)

class Transformer(nn.Module):
    """Full Transformer model: encoder, decoder, embeddings, positional encodings, projection."""
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
        """
        Args:
            src (Tensor): (batch_size, seq_length)
            src_mask (Tensor): (batch_size, 1, seq_length, seq_length)
        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
        embedded_src = self.src_embedding(src)  # (batch_size, seq_length, d_model)
        encoded_src = self.src_positional_encoding(embedded_src)  # (batch_size, seq_length, d_model)
        return self.encoder(encoded_src, src_mask)  # (batch_size, seq_length, d_model)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        Args:
            tgt (Tensor): (batch_size, seq_length)
            encoder_output (Tensor): (batch_size, seq_length, d_model)
            src_mask (Tensor): (batch_size, 1, seq_length, seq_length)
            tgt_mask (Tensor): (batch_size, 1, seq_length, seq_length)
        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
        embedded_tgt = self.tgt_embedding(tgt)  # (batch_size, seq_length, d_model)
        decoded_tgt = self.tgt_positional_encoding(embedded_tgt)  # (batch_size, seq_length, d_model)
        return self.decoder(decoded_tgt, encoder_output, src_mask, tgt_mask)  # (batch_size, seq_length, d_model)
    
    def project(self, decoder_output):
        return self.projection(decoder_output)  # (batch_size, seq_length, vocab_size)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):

    # model embeddings and positional encodings
    src_embedding = InputEmbedding(d_model,src_vocab_size)
    tgt_embedding = InputEmbedding(d_model,tgt_vocab_size)

    src_pos_enconding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_enconding = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder and decoder blocks
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

    # model projection
    projection = Projection(d_model, tgt_vocab_size)

    # the full transformer model
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos_enconding, tgt_pos_enconding, projection)

    # initialize parameters with Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
