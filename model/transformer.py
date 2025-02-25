import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=4):
        super(LoRALayer, self).__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)

    def forward(self, x):
        return self.up(self.down(x))

class MultiModalTransformer(nn.Module):
    def __init__(self, text_model, embedding_matrix):
        super(MultiModalTransformer, self).__init__()
        self.text_model = text_model
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.lora = LoRALayer(text_model.d_model, text_model.d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Process text input
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        # Apply LoRA
        src_emb = src_emb + self.lora(src_emb)
        tgt_emb = tgt_emb + self.lora(tgt_emb)

        # Pass through Transformer
        memory = src_emb
        for layer in self.text_model.encoder:
            memory = layer(memory, src_mask)
        output = tgt_emb
        for layer in self.text_model.decoder:
            output = layer(output, memory, tgt_mask)
        return self.text_model.generator(output)
