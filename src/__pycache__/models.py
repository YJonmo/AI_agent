
import torch
from PIL import Image
# No - this import statement has syntax errors and poor formatting. Here's the corrected version:
from utils import (
    cifar_transforms,
    cifar_model, 
    cifar_classes,
    pneumonia_transforms,
    pneumonia_model,
    pneumonia_classes
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pneumonia_model = pneumonia_model.to(device)
cifar_model = cifar_model.to(device)


def cifar_classifier(img:str='./data/horse.jpeg'):
    img = cifar_transforms(Image.open(img))
    img = img.to(device)
    cifar_class = torch.argmax(cifar_model(img))
    return cifar_classes[cifar_class]

def pneumonia_classifier(img:str='./data/person1_virus_6.jpeg'):
    img = pneumonia_transforms(Image.open(img))
    img = img.to(device)
    pneumonia_class = torch.argmax(pneumonia_model(img))
    return pneumonia_classes[pneumonia_class]


print(f"{device=}")
print(cifar_classifier())
print(pneumonia_classifier())