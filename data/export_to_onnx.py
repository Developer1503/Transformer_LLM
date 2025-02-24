import torch
from model.transformer import Transformer

def export_to_onnx(model, output_path):
    # Create a dummy input tensor with the same shape as your model's input
    dummy_input = torch.randint(0, 10000, (1, 50))  # Adjust the shape as needed
    torch.onnx.export(model, dummy_input, output_path, opset_version=11, input_names=['input'], output_names=['output'])

def main():
    # Load your trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000).to(device)
    model.load_state_dict(torch.load('path_to_your_model.pth', map_location=device))
    model.eval()

    # Export the model to ONNX format
    export_to_onnx(model, "model.onnx")

if __name__ == "__main__":
    main()
