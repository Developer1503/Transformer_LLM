import json
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor

class MultiModalDataset(Dataset):
    def __init__(self, data_path, tokenizer, processor, max_len):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        src = self.tokenizer.encode(item["caption"], max_length=self.max_len)
        tgt = self.tokenizer.encode(item["caption"], max_length=self.max_len)  # Example target
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.processor(images=image, return_tensors="pt")
        return torch.tensor(src), torch.tensor(tgt), image

    def __len__(self):
        return len(self.data)
