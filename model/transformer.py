import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class MultiModalTransformer(nn.Module):
    def __init__(self, text_model, vision_model_name="openai/clip-vit-base-patch32"):
        super(MultiModalTransformer, self).__init__()
        self.text_model = text_model
        self.vision_model = CLIPModel.from_pretrained(vision_model_name)
        self.processor = CLIPProcessor.from_pretrained(vision_model_name)

        # Fusion layer to merge text and vision features
        self.fusion_layer = nn.Linear(
            text_model.d_model + self.vision_model.config.hidden_size,
            text_model.d_model
        )

    def forward(self, src, tgt, image=None, src_mask=None, tgt_mask=None):
        # Process text input
        src_emb = self.text_model.embedding(src)
        tgt_emb = self.text_model.embedding(tgt)

        # Process image input
        if image is not None:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            image_features = self.vision_model.get_image_features(**inputs)
            image_features = image_features.unsqueeze(1).expand(-1, src_emb.size(1), -1)
            src_emb = torch.cat((src_emb, image_features), dim=-1)
            src_emb = self.fusion_layer(src_emb)

        # Pass through Transformer
        memory = src_emb
        for layer in self.text_model.encoder:
            memory = layer(memory, src_mask)
        output = tgt_emb
        for layer in self.text_model.decoder:
            output = layer(output, memory, tgt_mask)
        return self.text_model.generator(output)
