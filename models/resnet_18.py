import torch
import torchvision.models as models


def ResNet_18():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 10)
    return model
