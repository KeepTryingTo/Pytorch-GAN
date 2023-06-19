"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/14 11:06
"""

import os
from PIL import Image
from tqdm import tqdm

root_dir = 'data/FFHQ/images1024x1024'

for file in tqdm(os.listdir(root_dir)):
    img = Image.open(root_dir + '/' + file).resize(size=(128,128))
    img.save("FFHQ_resize/"+file)