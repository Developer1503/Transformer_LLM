# scripts/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Import the model from the app directory
from app.model import TransformerModel

# --- ⚙️ Hyperparameters ---
HPARAMS = {
    "corpus_path": "data/corpus.txt",
    "vocab_size": 8000,
    "val_split": 0.1,
    "window_size": 32,
    "batch_size": 128,
    "d_model": 256,
    "heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "epochs": 10,
    "lr": 1e-4,
}

# --- 1. Tokenizer Training ---
print("--- Starting Tokenizer Training ---")
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=HPARAMS["vocab_size"], special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"])
tokenizer.train([HPARAMS["corpus_path"]], trainer)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
vocab_size = tokenizer.get_vocab_size()
print(f"✅ Tokenizer trained with a vocabulary size of {vocab_size:,}")

# --- 2. Data Preparation ---
print("\n--- Preparing Datasets ---")
with open(HPARAMS["corpus_path"], "r", encoding="utf-8") as f:
    text = f.read()
seq = tokenizer.encode(text).ids

inputs, targets = [], []
for i in range(len(seq) - HPARAMS["window_size"]):
    inputs.append(seq[i : i + HPARAMS["window_size"]])
    targets.append(seq[i + 1 : i + HPARAMS["window_size"] + 1])

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, targets, test_size=HPARAMS["val_split"], random_state=42
)

class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.Y[i], dtype=torch.long)

train_dl = DataLoader(TextDataset(train_inputs, train_targets), batch_size=HPARAMS["batch_size"], shuffle=True)
val_dl = DataLoader(TextDataset(val_inputs, val_targets), batch_size=HPARAMS["batch_size"])
print(f"✅ Created {len(train_inputs):,} training samples and {len(val_inputs):,} validation samples.")

# --- 3. Model Training ---
print("\n--- Initializing Model Training ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerModel(
    vocab_size=vocab_size,
    d_model=HPARAMS["d_model"],
    heads=HPARAMS["heads"],
    d_ff=HPARAMS["d_ff"],
    num_layers=HPARAMS["num_layers"],
    dropout=HPARAMS["dropout"],
    max_len=HPARAMS["window_size"] + 1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

print(f"🚀 Starting training on {device} for {HPARAMS['epochs']} epochs...")
for epoch in range(HPARAMS["epochs"]):
    model.train()
    total_train_loss = 0
    for X, Y in train_dl:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        loss = loss_fn(pred.view(-1, pred.size(-1)), Y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dl)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X, Y in val_dl:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = loss_fn(pred.view(-1, pred.size(-1)), Y.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dl)
    
    print(f"Epoch {epoch+1:02}/{HPARAMS['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# --- 4. Save Artifacts ---
print("\n--- Saving Final Model and Tokenizer ---")
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/transformer_model.pth")
tokenizer.save("saved_models/tokenizer.json")
print("✅ Training complete and artifacts saved.")