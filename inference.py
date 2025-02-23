import torch
from model.transformer import Transformer
from data.tokenizer import SimpleTokenizer

def generate_response(model, src, tokenizer, device, max_len=50):
    model.eval()
    src = torch.tensor(tokenizer.encode(src, max_length=max_len)).unsqueeze(0).to(device)
    memory = model.encoder(model.src_embed(src))
    ys = torch.ones(1, 1).fill_(tokenizer.vocab['<sos>']).type_as(src.data).to(device)
    for _ in range(max_len - 1):
        out = model.decoder(model.tgt_embed(ys), memory)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == tokenizer.vocab['<eos>']:
            break
    return tokenizer.sequence_to_text(ys.cpu().numpy()[0])

def main():
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000).to(device)
    model.load_state_dict(torch.load('path_to_your_model.pth', map_location=device))
    tokenizer = SimpleTokenizer()

    # Interactive chat loop
    print("Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(model, user_input, tokenizer, device)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
