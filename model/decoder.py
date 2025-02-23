class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.sublayer[1](x + self.dropout(self.src_attn(x, memory, memory, src_mask)))
        x = self.sublayer[2](x + self.dropout(self.feed_forward(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
