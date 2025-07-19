import requests
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, Dataset
import streamlit as st

# --- Configuration & Model Definition ---

# ⚙️ Hyperparameters
HPARAMS = {
    "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
    "window_size": 16,
    "d_model": 128,
    "heads": 4,
    "d_ff": 512,
    "dropout": 0.1,
}

# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer('causal_mask', self.generate_square_subsequent_mask(max_len))

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        attn_output, _ = self.mha(x, x, x, attn_mask=self.causal_mask[:T, :T])
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return self.fc_out(x)

# --- Model Loading and Training ---

@st.cache_resource
def setup_model_and_vocab():
    device = "cpu"
    with st.spinner("Performing one-time model setup (scraping, training...). This may take a moment."):
        try:
            resp = requests.get(HPARAMS["url"])
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = " ".join(p.get_text() for p in soup.find_all("p"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error scraping URL: {e}")
            return None, None, None

        text = text.lower()
        text = re.sub(r'([\.])', r' \1 ', text)
        text = re.sub(r"[^a-z\. ]+", " ", text)
        tokens = text.split()

        local_vocab = {word: i + 2 for i, word in enumerate(sorted(list(set(tokens))))}
        local_vocab["<pad>"] = 0
        local_vocab["<unk>"] = 1
        local_inv_vocab = {i: w for w, i in local_vocab.items()}
        seq = [local_vocab.get(w, local_vocab["<unk>"]) for w in tokens]

        inputs, targets = [], []
        for i in range(len(seq) - HPARAMS["window_size"]):
            inputs.append(seq[i : i + HPARAMS["window_size"]])
            targets.append(seq[i + 1 : i + HPARAMS["window_size"] + 1])

        class TextDataset(Dataset):
            def __init__(self, X, Y): self.X, self.Y = X, Y
            def __len__(self): return len(self.X)
            def __getitem__(self, i): return torch.tensor(self.X[i]), torch.tensor(self.Y[i])

        dl = DataLoader(TextDataset(inputs, targets), batch_size=64, shuffle=True)

        model = TransformerEncoder(
            vocab_size=len(local_vocab),
            d_model=HPARAMS["d_model"],
            heads=HPARAMS["heads"],
            d_ff=HPARAMS["d_ff"],
            dropout=HPARAMS["dropout"],
            max_len=HPARAMS["window_size"] + 1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(5):
            for X, Y in dl:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                loss = loss_fn(pred.view(-1, pred.size(-1)), Y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    st.success("Model setup complete!")
    return model, local_vocab, local_inv_vocab

# --- Main App Logic ---

st.set_page_config(page_title="Live NLP Transformer", layout="centered")

st.title("Live NLP Transformer ✨")
st.markdown("Enter a prompt and let a simple Transformer model, trained live on the Wikipedia page for NLP, complete it for you.")

model, vocab, inv_vocab = setup_model_and_vocab()

if model:
    prompt = st.text_area("Your Prompt:", "natural language is a", height=100)

    if st.button("Generate Text", type="primary"):
        if prompt:
            with st.spinner("Generating..."):
                model.eval()
                tokens = [vocab.get(t, vocab["<unk>"]) for t in prompt.lower().split()]
                generated_text = prompt

                with torch.no_grad():
                    for _ in range(50):
                        input_seq = torch.tensor([tokens[-HPARAMS["window_size"]:]], device="cpu")
                        output = model(input_seq)
                        last_token_logits = output[0, -1, :]
                        next_token = torch.multinomial(F.softmax(last_token_logits, dim=-1), num_samples=1).item()
                        
                        if next_token == vocab["<pad>"]: break
                        
                        tokens.append(next_token)
                        generated_text += " " + inv_vocab.get(next_token, "<unk>")
                
                st.subheader("Generated Result:")
                st.markdown(f"> {generated_text}")
        else:
            st.warning("Please enter a prompt.")
else:
    st.error("Model could not be loaded. Please check the logs.")
