"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/4 16:17
"""
import torch
import albumentations
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/maps/maps/train"
VAL_DIR = "data/maps/maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 1
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "models/disc.pth.tar"
CHECKPOINT_GEN = "models/gen.pth.tar"

both_transform = albumentations.Compose(
    [albumentations.Resize(width=256, height=256),],
    additional_targets={"image0": "image"},
)

transform_only_input = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(p=0.2),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = albumentations.Compose(
    [
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)