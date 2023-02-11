import torch
import os

def get_selected_model(path, name):
    if name == "Multi-task EfficientNet-B2":
        return torch.load(os.path.join(path, 'enet_b0_8_best_afew.pt'), map_location=torch.device('cpu'))
    elif name == "FAN":
        return torch.load(os.path.join(path, 'fan_Resnet18_MS1M_pytorch.pt'), map_location=torch.device('cpu'))
    elif name == "InceptionResnetV2":
        return torch.load(os.path.join(path, 'model_43_75.2203_.pt'), map_location=torch.device('cpu'))