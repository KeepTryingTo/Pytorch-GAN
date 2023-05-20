"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/28 17:14
"""

import os
import torch
import config
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset

class vanGoPhotoDataset(Dataset):
    def __init__(self,root_vango,root_photo,transform=None):
        super(vanGoPhotoDataset, self).__init__()
        self.root_vango = root_vango
        self.root_photo = root_photo
        self.transform = transform

        self.vango_Images = os.listdir(self.root_vango)
        self.photo_Images = os.listdir(self.root_photo)

        self.length_dataset = max(len(self.vango_Images),len(self.photo_Images))
        self.vango_len = len(self.vango_Images)
        self.photo_len = len(self.photo_Images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        vango_img = self.vango_Images[index % self.vango_len]
        photo_img = self.photo_Images[index % self.photo_len]

        vango_path = os.path.join(self.root_vango,vango_img)
        photo_path = os.path.join(self.root_photo,photo_img)

        vango_img = np.array(Image.open(vango_path).convert("RGB"))
        photo_img = np.array(Image.open(photo_path).convert("RGB"))

        if self.transform:
            argumentation = self.transform(image = vango_img,image0 = photo_img)
            vango_img = argumentation["image"]
            photo_img = argumentation["image0"]
        return vango_img,photo_img


