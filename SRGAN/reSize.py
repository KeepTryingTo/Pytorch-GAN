"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/16 9:51
"""

import os
import cv2

def resize(imgsPath = 'test_images/'):
    imgs = os.listdir(imgsPath)
    for index,imgName in enumerate(imgs):
        imgPath = os.path.join(imgsPath,imgName)
        print(imgPath)
        img = cv2.imread(imgPath)
        img = cv2.resize(img,(24,24),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(imgsPath,imgName),img)
    print('done......')

if __name__ == '__main__':
    resize()

