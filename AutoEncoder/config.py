"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/6 16:23
"""
import torch
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 100
BEATAS=(0.5,0.999)
TRAIN_DATA = 'data/FashionMNIST'
VAL_DATA = 'data/FashionMNIST'
MODEL = "models/AE.pth.tar"
SAVE_CHECKPOINT=True
LOAD_CHECKPOINT=False

transform = transforms.Compose([
    transforms.Resize(size = (28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5]),
])


