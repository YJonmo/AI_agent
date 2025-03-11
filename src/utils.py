from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import transforms

root_dir = Path(__file__).parent.parent.resolve()


cifar_classes = ('plane', 'car', 'bird', 'cat', 'deed', 
           'dog', 'frog', 'horse', 'ship', 'truck')
pneumonia_classes = ('pneumonia not identidied', 'pneumonia')

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pneumonia_model = torch.jit.load(str(weight_path/'pneumonia.pt'), map_location=device).eval()
cifar_model = torch.jit.load(str(weight_path/'cifar_res18.pt'), map_location=device).eval()# Yes, we can run CUDA-trained models on CPU by mapping them to CPU device
