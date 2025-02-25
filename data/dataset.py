import json
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        src = self.tokenizer.encode(item["text"], max_length=self.max_len)
        tgt = self.tokenizer.encode(item["text"], max_length=self.max_len)  # Example target
        return torch.tensor(src), torch.tensor(tgt)

    def __len__(self):
        return len(self.data)
