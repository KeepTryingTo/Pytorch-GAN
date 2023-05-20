"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/29 11:36
"""
import torch
import albumentations #深度学习增强库
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/vangogh2photo/train"
VAL_DIR = "data/vangogh2photo/val"

BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H="gen.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITICH_H="critich.pth.tar"
CHECKPOINT_CRITICH_Z="criticz.pth.tar"

transform = albumentations.Compose(
    [
        albumentations.Resize(width=256,height=256),
        albumentations.HorizontalFlip(p = 0.5),
        albumentations.ColorJitter(p=0.1),#颜色抖动
        albumentations.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        albumentations.pytorch.ToTensorV2()
    ],
    additional_targets={"image0":"image"},
)
