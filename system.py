# app.py - Streamlit App for the NLP Transformer Model
# Run it with: streamlit run app.py
!pip install -q streamlit

import requests
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, Dataset
import streamlit as st

# --- Configuration ---
HPARAMS = {
    "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
    "window_size": 16,
    "d_model": 128,
    "heads": 4,
    "d_ff": 512,
    "dropout": 0.1,
}

# --- Model Components ---
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
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads, d_ff, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer('causal_mask', self._generate_causal_mask(max_len))

    def _generate_causal_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        attn_output, _ = self.mha(x, x, x, attn_mask=self.causal_mask[:T, :T])
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.dropout(self.ff(x)))
        return self.fc_out(x)

# --- Model Training & Caching ---
@st.cache_resource
def setup_model_and_vocab():
    device = "cpu"
    try:
        resp = requests.get(HPARAMS["url"])
        soup = BeautifulSoup(resp.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception as e:
        st.error(f"Failed to fetch Wikipedia content: {e}")
        return None, None, None

    text = text.lower()
    text = re.sub(r'([\.])', r' \1 ', text)
    text = re.sub(r"[^a-z\. ]+", " ", text)
    tokens = text.split()

    vocab = {word: i + 2 for i, word in enumerate(sorted(set(tokens)))}
    vocab["<pad>"], vocab["<unk>"] = 0, 1
    inv_vocab = {i: w for w, i in vocab.items()}
    seq = [vocab.get(w, vocab["<unk>"]) for w in tokens]

    inputs, targets = [], []
    for i in range(len(seq) - HPARAMS["window_size"]):
        inputs.append(seq[i:i + HPARAMS["window_size"]])
        targets.append(seq[i + 1:i + HPARAMS["window_size"] + 1])

    class TextDataset(Dataset):
        def __init__(self, X, Y): self.X, self.Y = X, Y
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return torch.tensor(self.X[i]), torch.tensor(self.Y[i])

    dl = DataLoader(TextDataset(inputs, targets), batch_size=64, shuffle=True)

    model = TransformerEncoder(
        vocab_size=len(vocab),
        d_model=HPARAMS["d_model"],
        heads=HPARAMS["heads"],
        d_ff=HPARAMS["d_ff"],
        dropout=HPARAMS["dropout"],
        max_len=HPARAMS["window_size"] + 1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(5):
        for X, Y in dl:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = loss_fn(pred.view(-1, pred.size(-1)), Y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, vocab, inv_vocab

# --- Streamlit UI ---
st.set_page_config(page_title="NLP Transformer", layout="centered")
st.title("🧠 Mini NLP Transformer (Live Training)")
st.markdown("A simple Transformer encoder trained on the Wikipedia article for NLP. Try prompting it below.")

model, vocab, inv_vocab = setup_model_and_vocab()

with st.sidebar:
    st.header("⚙️ Options")
    max_tokens = st.slider("Tokens to generate", 10, 100, 50)
    st.markdown("---")
    st.markdown("Built with 💡 Streamlit + PyTorch")

if model:
    prompt = st.text_area("💬 Prompt:", "natural language is", height=100)

    if st.button("🚀 Generate"):
        if prompt.strip():
            model.eval()
            tokens = [vocab.get(w, vocab["<unk>"]) for w in prompt.lower().split()]
            generated = tokens.copy()

            with torch.no_grad():
                for _ in range(max_tokens):
                    input_seq = torch.tensor([generated[-HPARAMS["window_size"]:]], dtype=torch.long)
                    output = model(input_seq)
                    last_logits = output[0, -1]
                    next_token = torch.multinomial(F.softmax(last_logits, dim=-1), 1).item()
                    if next_token == vocab["<pad>"]: break
                    generated.append(next_token)

            result = " ".join([inv_vocab.get(t, "<unk>") for t in generated])
            st.subheader("📝 Generated Output")
            st.markdown(f"```\n{result}\n```")
        else:
            st.warning("Prompt can't be empty.")
else:
    st.error("Model initialization failed.")
