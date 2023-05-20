"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/17 17:17
"""

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_DIR = "data"
LEARNING_RATIO = 1e-4
BATCH_SIZE = 16
BEAT1 = 0.9
BEAT2 = 0.9
EPSILON = 1e-5
SHUFFLE = True
NUM_WORKERS = 0
SAVE_MODELS = "models"
LOAD_MODELS = "models"
SAVE_IMAGES = "images"
NUM_EPOCHS = 1
PIN_MEMEORY = False
LOSS_RATIO = 1e-3