"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/29 14:34
"""

import cv2
import time
import torch
import config
import tkinter
import numpy as np
import albumentations
from PIL import Image,ImageTk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from torchvision import transforms
from net.Generator import Generator
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2

transform = transforms.Compose(
    [
        transforms.Resize(size=(256,256)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ColorJitter(brightness=0.2),#颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ]
)

#显示的图片大小
img_width = 200
img_height = 200

#创建主窗口
root = tkinter.Tk()
root.title('DCGAN随机生成图片')
root.geometry("560x560")
#root.iconphoto(True,tkinter.PhotoImage(file = 'hometown.png'))


def Canvas_(root):
    # 创建画布
    canvas0 = tkinter.Canvas(root, bg='white', width=img_width, height=img_height)
    canvas0.place(x=50, y=50)
    # 创建画布
    canvas1 = tkinter.Canvas(root, bg='white', width=img_width, height=img_height)
    canvas1.place(x=300, y=50)
    # 创建标签
    label1 = tkinter.Label(root, text='选择自然景 or 梵高风格图', font=('黑体', 13), width=30, height=1)
    # `anchor=nw`则是把图片的左上角作为锚定点
    label1.place(x=15, y=20, anchor='nw')
    # 创建标签
    label2 = tkinter.Label(root, text='梵高风格 or 自然图', font=('黑体', 13), width=25, height=1)
    # `anchor=nw`则是把图片的左上角作为锚定点
    label2.place(x=280, y=20, anchor='nw')
    return [canvas0,canvas1]

def showImage(frame):
    """
    :param root: 主窗口
    :return:
    """
    # 摄像头翻转
    # frame=cv2.flip(frame,1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    image = image.astype(np.uint8)
    PILimage = Image.fromarray(image)
    PILimage = PILimage.resize((img_width, img_height), Image.ANTIALIAS)
    try:
        tkImage = ImageTk.PhotoImage(image=PILimage)
    except:
        return 0
    return tkImage


#这里需要将tkImage设置为全局变量，不然显示不图片
tkImage1=''
tkImage2=''
def set_BackGround(image_path1,image_path2, root):
    global tkImage1
    global tkImage2
    canvas0,canvas1 = Canvas_(root)
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    tkImage1 = showImage(img1)
    tkImage2 = showImage(img2)
    canvas0.create_image(0, 0, anchor='nw', image=tkImage1)
    canvas1.create_image(0, 0, anchor='nw', image=tkImage2)

#选择要生成的图像文件
imagePath={"filename": ""}#保存点击事件之后的图片路径
tkImage = ''
def selectFilename(root):
    # 选择open文件进行识别
    global tkImage
    #注意：选择的图片路径中不要包含中文，不然OpenCV读取图片的时候读取为空
    image_path1 = filedialog.askopenfilename(title='选择文件进行识别')
    print(image_path1)
    if image_path1:
        canvas0, canvas1 = Canvas_(root)
        img1 = cv2.imread(image_path1)
        print(type(img1))
        tkImage = showImage(img1)
        canvas0.create_image(0, 0, anchor='nw', image=tkImage)
        imagePath["filename"] = image_path1
        print(imagePath["filename"])
    else:
        tkinter.messagebox.showwarning(title='警告',message='请选择文件')

#关闭主界面时弹出的确认窗口
def closeEvent(root):
    # root.withdraw()  # 仅显示对话框，隐藏主窗口
    btn_close=tkinter.Button(root,text='退出',font=('黑体', 14), height=1, width=15,command=root.destroy)
    btn_close.place(x=350,y=450)
    return True

def generagedImage(model):
    """
    :param model:
    :return:
    """
    img = Image.open(imagePath["filename"]).convert("RGB")
    img_transform = transform(img)
    img_transform = torch.unsqueeze(input=img_transform,dim=0)
    transformImg = model(img_transform)
    save_image(transformImg * 0.5 + 0.5,f"cur.png")

tkImage0 = ''
def loadModel(model,root):
    """
    :param model:
    :return:
    """
    # 注意这里的变量，一定要声明为全局变量，不然后面显示不出来图片
    global  tkImage0
    #生成图片
    generagedImage(model)
    time.sleep(0.5)
    #从保存的生成器生成图片中读取该图片并显示出来
    canvas0,canvas1 = Canvas_(root)
    img = cv2.imread('cur.png')
    tkImage0 = showImage(img)
    canvas1.create_image(0, 0, anchor='nw', image=tkImage0)


#点击生成图片按钮
def ButtonImage(model1,model2,root):
    # root.withdraw()  # 仅显示对话框，隐藏主窗口
    btn_gen0 = tkinter.Button(root, text='选择梵高 or 自然景', font=('黑体', 14), height=1, width=20,
                              command = lambda : selectFilename(root))
    btn_gen0.place(x=20, y=355)

    btn_gen1 = tkinter.Button(root, text='生成梵高风格', font=('黑体', 14), height=1, width=20,
                               command = lambda : loadModel(model1,root))
    btn_gen1.place(x=300, y=330)
    btn_gen2 = tkinter.Button(root, text='生成自然图', font=('黑体', 14), height=1, width=20,
                             command = lambda : loadModel(model2,root))
    btn_gen2.place(x=300, y=370)


if __name__ == '__main__':
    #image_path是一张背景图
    image_path1 = 'cat.png'
    image_path2 = 'cat1.png'
    set_BackGround(image_path1=image_path1,image_path2 = image_path2,root = root)
    #这里的模型由于是在GPU上训练的， 所以加载方式需要改变一下
    #gen_G:将自然景转换为梵高风格 gen_F: 将梵高风格转换为自然景
    model1 = Generator(img_channels=3)
    checkpoint = torch.load('models/gen_G.pth.tar',map_location=lambda storage, loc: storage)
    model1.load_state_dict(checkpoint['state_dict'])

    model2 = Generator(img_channels=3)
    checkpoint = torch.load('models/gen_F.pth.tar', map_location=lambda storage, loc: storage)
    model2.load_state_dict(checkpoint['state_dict'])
    while True:
        ButtonImage(model1,model2,root)
        flag = closeEvent(root)
        if flag:
            break
    root.mainloop()