import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        src = self.tokenizer.encode(item['src'], max_length=self.max_len, padding='post')
        tgt = self.tokenizer.encode(item['tgt'], max_length=self.max_len, padding='post')
        return torch.tensor(src), torch.tensor(tgt)

def load_data(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_data(data, tokenizer, max_len):
    dataset = ChatDataset(data, tokenizer, max_len)
    return dataset
