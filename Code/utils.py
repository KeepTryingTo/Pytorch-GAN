"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/4/29 11:27
"""

import copy
import os
import random

import torch
import config
import numpy as np

def save_checkpoint(model,optimizer,filename = "my_checkpoint.pth.tar",epochs = 0):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    torch.save(checkpoint,filename + str(epochs))
def load_checkpoin(checkpoint_file,model,optimizer,lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file,map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_group:
        param_group["lr"] = lr

def seed_everthing(seed = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False



if __name__ == '__main__':
    pass
