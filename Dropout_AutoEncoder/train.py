"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/6 15:53
"""
import time
import torch
import config
import utils
from net.AE import AE
from tqdm import tqdm
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader

#下载数据集
trainDataset = torchvision.datasets.FashionMNIST(root=config.TRAIN_DATA,train=True,
                                          transform=config.transform,download=True)
valDataset = torchvision.datasets.FashionMNIST(root=config.VAL_DATA,train=False,
                                          transform=config.transform,download=True)

#加载数据集
trainLoader = DataLoader(
    dataset=trainDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
valLoader = DataLoader(
    dataset=valDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

#加载模型
AE_Model = AE(in_feautres=784,out_features=128).to(config.DEVICE)

#定义损失函数和优化器
loss_fn = torch.nn.MSELoss()

opt_AE = torch.optim.Adam(AE_Model.parameters(),lr = config.LEARNING_RATE,betas=(0.9,0.999))

#保存损失值
loss_ = []

#加载预训练模型
if config.LOAD_CHECKPOINT:
    utils.load_checkPoint(model=AE_Model,filename=config.MODEL,optimizer=opt_AE,lr=config.LEARNING_RATE)

for epoch in range(config.NUM_EPOCHS):
    step_loss = 0
    dataLen = len(trainLoader)
    AE_Model.train()
    loop = tqdm(trainLoader,leave=True)
    loop.set_description(desc="training: ")
    for step ,data in enumerate(loop):
        imgs,labels = data
        imgs,labels = imgs.to(config.DEVICE),labels.to(config.DEVICE)
        #encoder=>decoder
        decoder_img = AE_Model(imgs)
        loss = loss_fn(decoder_img,imgs)
        opt_AE.zero_grad()
        loss.backward()
        opt_AE.step()

        loop.set_description(desc="training: ")
        step_loss += loss.item()
        if step % 10 == 0 and step > 0:
            loop.set_postfix(epoch = epoch,loss = loss.item())
            # print("---------------------------Loss: {:.6f}----------------------------".format(loss))
    loss_.append(step_loss / dataLen)
    print("---------------------------Loss: {:.6f}----------------------------".format(step_loss / dataLen))
    time.sleep(0.3)
    if epoch % 10 == 0 and epoch > 0:
        with torch.no_grad():
            loop = tqdm(iterable=valLoader,leave=True)
            for step,data in enumerate(loop):
                imgs,labels = data
                if step % 100 == 0 and step > 0:
                    utils.save_epoch_image(model=AE_Model,filename="images",val_loader=imgs,epoch = epoch)

utils.save_checkPoint(model=AE_Model,optimizer=opt_AE,filename=config.MODEL)
utils.draw(loss_,config.NUM_EPOCHS)