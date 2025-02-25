import torch
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor
from data.tokenizer import SimpleTokenizer
from model.transformer import MultiModalTransformer

def predict(input_text, image_path=None):
    # Load Tokenizer & ONNX Model
    tokenizer = SimpleTokenizer()
    ort_session = ort.InferenceSession("model_pruned.onnx")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process Image (if provided)
    if image_path:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)

    # Tokenize Text
    src = torch.tensor(tokenizer.encode(input_text, max_length=50)).unsqueeze(0).to(device)

    # Prepare ONNX inputs
    ort_inputs = {"input": src.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Decode Output
    output_text = tokenizer.sequence_to_text(ort_outs[0][0])
    return output_text

# Run Inference
if __name__ == "__main__":
    text_input = "Describe this image"
    image_path = "example.jpg"
    result = predict(text_input, image_path)
    print("Response:", result)
