import tensorflow
import torch, torchvision

mpsDevice = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torchvision.models.resnet50()
model_mps = model.to(device=mpsDevice)
sample_input = torch.randn((32, 3, 254, 254), device=mpsDevice)
prediction = model_mps(sample_input)
print(prediction)