"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 13:13
"""
import torch

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.BatchNorm2d)):
            torch.nn.init.normal(m.weight.data,0.0,0.02)
