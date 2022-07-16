import torch
import torchvision.models as models


def ResNet_18():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 10)
    return model
