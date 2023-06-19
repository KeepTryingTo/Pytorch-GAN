"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/16 9:40
"""

import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))
            # print('self.data: {}'.format(self.data))
            # break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        #得到图像的完整路径
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        #将读取的图像转换为numpy形式
        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        #对图像进行预处理
        image = config.both_transforms(image=image)["image"]
        #对高分辨图像进行处理
        high_res = config.highres_transform(image=image)["image"]
        #将[96 x 96] => [24 x 24]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir=r"data")
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()