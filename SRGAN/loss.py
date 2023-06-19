"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/16 9:39
"""
import config
import torch.nn as nn
from torchinfo import summary
from torchvision.models import vgg19


# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #features map = [1, 512, 14, 14]
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        print(self.vgg)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

if __name__ == '__main__':
    # vgg = vgg19(pretrained = True)
    # print(vgg)
    # summary(vgg,input_size=(1,3,224,224))
    pass