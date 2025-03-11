from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import transforms

root_dir = Path(__file__).parent.resolve()


cifar_classes = ('plane', 'car', 'bird', 'cat', 'deed', 
           'dog', 'frog', 'horse', 'ship', 'truck')
pneumonia_classes = ('normal', 'pneumonia')

cifar_transforms = transforms.Compose([
                                        transforms.Lambda(lambda x: x.convert('RGB')),
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5)),
                                        transforms.Lambda(lambda x: x.unsqueeze(0))
                                       ])

pneumonia_transforms = transforms.Compose([
                                        transforms.Lambda(lambda x: x.convert('L')),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                                mean=[0.456],
                                                std=[0.224]
                                        ),
                                        transforms.Lambda(lambda x: x.unsqueeze(0))
                                    ])


weight_path = root_dir / 'weights'
pneumonia_model = torch.jit.load(str(weight_path/'pneumonia.pt')).eval()
cifar_model = torch.jit.load(str(weight_path/'cifar_res18.pt')).eval()
