import torch.nn as nn
from model.attention import SlidingWindowAttention
from utils.positional_encoding import RelativePositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, window_size=512):
        super(Transformer, self).__init__()
        self.embedding_size = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = RelativePositionalEncoding(d_model, dropout)

        self_attn = SlidingWindowAttention(d_model, nhead, window_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.pos_encoder(self.src_embed(src))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt))

        # Use checkpointing for encoder and decoder
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        return self.generator(output)
