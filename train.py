from torch.cuda.amp import autocast

def train(model, data_loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    epoch_loss = 0
    for src, tgt, image in data_loader:
        src, tgt = src.to(device), tgt.to(device)
        image = image.to(device) if image is not None else None

        optimizer.zero_grad()
        with autocast():
            output = model(src, tgt[:, :-1], image)
            loss = criterion(output.view(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    return epoch_loss
