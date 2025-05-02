import torch

print(torch.__version__)  # 输出版本，如 2.1.0

# 检查 GPU 支持（NVIDIA）
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    # 测试 GPU 计算
    x = torch.tensor([1.0]).cuda()
    print(x)  # 输出 tensor([1.0], device='cuda:0')
else:
    print("未检测到 GPU，使用 CPU 版本")

# Apple Silicon 专用检查
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS 加速已启用")
    y = torch.tensor([1.0], device="mps")
    print(y)
