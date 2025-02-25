from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor

class MultiModalDataset(Dataset):
    def __init__(self, data, tokenizer, processor, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        src = self.tokenizer.encode(item["text"], max_length=self.max_len)
        tgt = self.tokenizer.encode(item["target"], max_length=self.max_len)
        image = Image.open(item["image_path"]).convert("RGB") if "image_path" in item else None
        image = self.processor(images=image, return_tensors="pt") if image else None
        return torch.tensor(src), torch.tensor(tgt), image

    def __len__(self):
        return len(self.data)
