import torch
import onnx

# 指定输入和输出的名称
input_name = 'input'   # 根据实际情况调整
output_name = 'output' # 根据实际情况调整

# 加载已经训练完成的 ONNX 模型
model_path = './logs/model6.onnx'
model = onnx.load(model_path)

# 导出新的 ONNX 模型
new_model_path = './logs/model7.onnx'
dummy_input = torch.randn(1, 3, 224, 224)  # 根据模型输入大小调整

# 重新导出为 model7.onnx
torch.onnx.export(
    model,
    dummy_input,
    new_model_path,
    opset_version=13,  # 注意版本选择
    input_names=[input_name]
)

print(f"Model exported as: {new_model_path}")
