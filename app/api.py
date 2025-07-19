# app/api.py

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

# Import the model architecture from the model.py file
from app.model import TransformerModel
from tokenizers import Tokenizer

# --- 🚀 API Initialization ---
print("--- Initializing Flask App and Loading Models ---")
app = Flask(__name__)

# Config - should match training HPARAMS
HPARAMS = {
    "vocab_size": 8000,
    "window_size": 32,
    "d_model": 256,
    "heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
}

# Use CPU for inference
device = torch.device("cpu")

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("saved_models/tokenizer.json")

# Initialize and load the model
model = TransformerModel(
    vocab_size=HPARAMS["vocab_size"],
    d_model=HPARAMS["d_model"],
    heads=HPARAMS["heads"],
    d_ff=HPARAMS["d_ff"],
    num_layers=HPARAMS["num_layers"],
    dropout=HPARAMS["dropout"],
    max_len=HPARAMS["window_size"] + 1
).to(device)

model.load_state_dict(torch.load("saved_models/transformer_model.pth", map_location=device))
model.eval()
print("✅ Model and Tokenizer loaded successfully!")

def generate_text_from_prompt(prompt, length=50):
    """Generates text from a prompt, returning the full text."""
    tokens = tokenizer.encode(prompt.lower()).ids
    
    with torch.no_grad():
        for _ in range(length):
            input_seq_ids = tokens[-HPARAMS["window_size"]:]
            input_tensor = torch.tensor([input_seq_ids], device=device, dtype=torch.long)
            
            output = model(input_tensor)
            last_token_logits = output[0, -1, :]
            
            # Use top-k sampling for better quality
            top_k_logits, top_k_indices = torch.topk(last_token_logits, k=50, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_index].item()
            
            if next_token == tokenizer.token_to_id("<eos>"): break
            tokens.append(next_token)

    return tokenizer.decode(tokens)

# Define the API endpoint
@app.route("/generate", methods=["POST"])
def handle_generation():
    """Handles POST requests to generate text."""
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = request.json['prompt']
    length = request.json.get('length', 50) 

    generated_text = generate_text_from_prompt(prompt, length)
    
    return jsonify({
        "prompt": prompt,
        "response": generated_text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)