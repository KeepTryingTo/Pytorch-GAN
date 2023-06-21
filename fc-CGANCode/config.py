"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/19 10:05
"""

import torch
from torchvision import transforms

DATASET = 'data/'
CHECKPOINT_GEN = 'models/gen.pth.tar'
CHECKPOINT_DISC = 'models/disc.pth.tar'
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARING_RATIO = 0.0002
NUM_WORKS = 0
BETA1 = 0.5
BETA2 = 0.999
LATENT_DIM = 100
NUM_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1
DROP_LAST = True
SHUFFLE = True
SAMPLE_INTERVAL = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform=transforms.Compose([
               transforms.Resize((IMG_SIZE,IMG_SIZE)),
               transforms.ToTensor(),
               transforms.Normalize((0.5), (0.5))
           ])