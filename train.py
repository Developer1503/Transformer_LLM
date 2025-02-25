import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from data.dataset import TextDataset
from data.tokenizer import MultiModalTokenizer
from model.transformer import MultiModalTransformer

def train(model, data_loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    epoch_loss = 0
    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        with autocast():
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    return epoch_loss

def main():
    # Hyperparameters
    d_model = 512
    num_heads = 8
    num_layers = 6
    batch_size = 32
    lr = 0.0001
    epochs = 20
    src_vocab_size = 10000  # Example vocab size
    tgt_vocab_size = 10000  # Example vocab size
    max_len = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer with pre-trained embeddings
    tokenizer = MultiModalTokenizer(embedding_matrix=np.random.rand(src_vocab_size, d_model))

    # Load dataset
    dataset = TextDataset("scraped_text_data.json", tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    text_model = DummyTextModel(d_model, src_vocab_size)  # Replace with your actual text model
    model = MultiModalTransformer(text_model, tokenizer.embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training loop
    for epoch in range(epochs):
        loss = train(model, data_loader, optimizer, criterion, device, epoch, scaler)

if __name__ == "__main__":
    main()
