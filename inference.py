import onnxruntime as ort
import torch
import numpy as np
from data.tokenizer import SimpleTokenizer
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer, util

class MultiModalTransformer(nn.Module):
    def __init__(self, text_model, vision_model_name='openai/clip-vit-base-patch32'):
        super(MultiModalTransformer, self).__init__()
        self.text_model = text_model
        self.vision_model = CLIPModel.from_pretrained(vision_model_name)
        self.processor = CLIPProcessor.from_pretrained(vision_model_name)
        self.fusion_layer = nn.Linear(text_model.d_model + vision_model.config.hidden_size, text_model.d_model)

    def forward(self, src, tgt, image=None, src_mask=None, tgt_mask=None):
        # Process text input
        src_emb = self.text_model.embedding(src)
        tgt_emb = self.text_model.embedding(tgt)

        # Process image input
        if image is not None:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            image_features = self.vision_model.get_image_features(**inputs)
            image_features = image_features.unsqueeze(1).expand(-1, src_emb.size(1), -1)
            src_emb = torch.cat((src_emb, image_features), dim=-1)
            src_emb = self.fusion_layer(src_emb)

        memory = src_emb
        for layer in self.text_model.encoder:
            memory = layer(memory, src_mask)
        output = tgt_emb
        for layer in self.text_model.decoder:
            output = layer(output, memory, tgt_mask)
        return self.text_model.generator(output)

class RAGPipeline:
    def __init__(self, model, retriever_model_name='all-MiniLM-L6-v2'):
        self.model = model
        self.retriever = SentenceTransformer(retriever_model_name)
        self.knowledge_base = ["Example document 1", "Example document 2"]  # Replace with your knowledge base

    def retrieve(self, query, top_k=5):
        query_emb = self.retriever.encode(query)
        doc_emb = self.retriever.encode(self.knowledge_base)
        scores = util.pytorch_cos_sim(query_emb, doc_emb)[0]
        top_results = torch.topk(scores, top_k)
        return [self.knowledge_base[idx] for idx in top_results[1]]

    def generate(self, input_text):
        retrieved_docs = self.retrieve(input_text)
        augmented_input = input_text + " ".join(retrieved_docs)
        # Generate response using the model
        src = torch.tensor(self.model.tokenizer.encode(augmented_input, max_length=50)).unsqueeze(0)
        tgt = torch.tensor([[self.model.tokenizer.vocab['<sos>']]])
        output = self.model(src, tgt)
        response = self.model.tokenizer.sequence_to_text(output[0].argmax(dim=-1).tolist())
        return response

def predict(input_text, ort_session, tokenizer, device, model, image=None):
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
    ort_session = ort.InferenceSession("model_pruned.onnx")
    tokenizer = SimpleTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    model = MultiModalTransformer(Transformer(src_vocab_size=10000, tgt_vocab_size=10000))
    rag_pipeline = RAGPipeline(model)

    # Interactive chat loop
    print("Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = rag_pipeline.generate(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
