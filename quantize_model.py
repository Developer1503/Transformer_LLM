import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load the ONNX model
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Quantize the model
quantized_model_path = "model_quantized.onnx"
quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)

print(f"Quantized model saved at {quantized_model_path}")
