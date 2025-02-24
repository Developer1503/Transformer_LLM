import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.onnx_model_bakery import quantize_weights, prune_model

# Load the ONNX model
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Quantize the model
quantized_model_path = "model_quantized.onnx"
quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)

# Prune the model
pruned_model_path = "model_pruned.onnx"
prune_model(quantized_model_path, pruned_model_path, sparsity_ratio=0.5)

print(f"Quantized and pruned model saved at {pruned_model_path}")
