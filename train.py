import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.transformer import Transformer
from data.dataset import ChatDataset, load_data, preprocess_data
from data.tokenizer import SimpleTokenizer
import os

def train(model, data_loader, optimizer, criterion, device, epoch):
    model.train()
    epoch_loss = 0
    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    return avg_loss

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

    # Initialize model, optimizer, and loss function
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Load data
    tokenizer = SimpleTokenizer()
    data = load_data('path_to_your_data.json')
    dataset = preprocess_data(data, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        loss = train(model, data_loader, optimizer, criterion, device, epoch)

        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

if __name__ == "__main__":
    main()
