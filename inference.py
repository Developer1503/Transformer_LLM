import onnxruntime as ort
import torch
import numpy as np
from data.tokenizer import SimpleTokenizer

def predict(input_text, ort_session, tokenizer, device):
    # Tokenize input text
    src = torch.tensor(tokenizer.encode(input_text, max_length=50)).unsqueeze(0).to(device)
    # Prepare input for ONNX Runtime
    ort_inputs = {"input": src.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    # Process the output to generate the response
    output_text = tokenizer.sequence_to_text(ort_outs[0][0])
    return output_text

def main():
    # Load quantized ONNX model
    ort_session = ort.InferenceSession("model_quantized.onnx")
    tokenizer = SimpleTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Interactive chat loop
    print("Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = predict(user_input, ort_session, tokenizer, device)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
