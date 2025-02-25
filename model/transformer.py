import torch
import torch.nn as nn

class MultiModalTransformer(nn.Module):
    def __init__(self, text_model):
        super(MultiModalTransformer, self).__init__()
        self.text_model = text_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Process text input
        src_emb = self.text_model.embedding(src)
        tgt_emb = self.text_model.embedding(tgt)

        # Pass through Transformer
        memory = src_emb
        for layer in self.text_model.encoder:
            memory = layer(memory, src_mask)
        output = tgt_emb
        for layer in self.text_model.decoder:
            output = layer(output, memory, tgt_mask)
        return self.text_model.generator(output)
