"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/17 16:45
"""
import torch
from torchinfo import summary

class Encoder(torch.nn.Module):
    def __init__(self,hidden_dim = 512,latent_dim = 2):
        super(Encoder, self).__init__()
        self.initial_dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=784, out_features=hidden_dim),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features = hidden_dim,out_features=256),
            torch.nn.ReLU(inplace=True)
        )
        #输出的均值和方差
        self.z_mean = torch.nn.Linear(in_features = 256,out_features=latent_dim)
        self.z_log_var = torch.nn.Linear(in_features=256,out_features=latent_dim)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.initial_dense(x)

        z_mean =  self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean,z_log_var

class Decoder(torch.nn.Module):
    def __init__(self,hidden_dim = 256,latent_dim = 2,num_features = 784):
        super(Decoder, self).__init__()
        self.initial_dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_dim,out_features=hidden_dim),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=hidden_dim,out_features=hidden_dim * 2),
            torch.nn.ReLU(inplace=True)
        )

        self.imgs = torch.nn.Linear(in_features=hidden_dim * 2,out_features=num_features)

    def forward(self,x):
        x = self.initial_dense(x)
        imgs = self.imgs(x)
        imgs = imgs.view(-1,28,28)
        return imgs


if __name__ == '__main__':
    x = torch.randn(size=(1,28,28))
    encoder = Encoder(hidden_dim=512,latent_dim=2)
    summary(encoder,input_size=(784,))
    z_mean,z_log_var = encoder(x)
    print("z_mean.shape: {}------z_log_var.shape: {}".format(z_mean.shape,z_log_var.shape))

    decoder = Decoder(hidden_dim=256,latent_dim=2,num_features=784)
    summary(decoder,input_size=(2,))
    y = torch.randn(size=(1,2))
    print("imgs.shape: {}".format(decoder(y).shape))


