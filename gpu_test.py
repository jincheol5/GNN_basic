import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. CPU will be used.")