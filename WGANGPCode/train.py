"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 12:06
"""
import torch
import torchvision
from draw import DrawGen
from torchvision import transforms
from utils import gradient_penality
from net import Discriminator,Generator
from initialize import initialize_weights
from torch.utils.data import DataLoader,dataset
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义相关参数
LEARNING_RATE = 1e-4#5e-5
BATCH_SIZE = 16
IMAGE_SIZE = 64
CHANNELS_IMG = 3#1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
WEIGHT_CLIP=0.01
LAMBDA_GP=10

transform = transforms.Compose([
    transforms.Resize(size = (IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5 for _ in range(CHANNELS_IMG)],
        std=[0.5 for _ in range(CHANNELS_IMG)]
    )
])

#下载数据集
#注意这里如果使用MNIST数据集，那么通道数为1，相应的地方需要将通道数修改为1
# dataset = datasets.MNIST(
#     root='data/MNIST',train=True,transform=transform,download=True
# )
dataset = torchvision.datasets.ImageFolder(
    root=r"E:\conda_3\PyCharm\Transer_Learning\WGAN\WGANCode\data",
    transform=transform
)
#加载数据集
dataLoader = DataLoader(
    dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0
)

#实例模型
gen = Generator.Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
#这里的判别模型使用cirtic，主要是区别于之前的discriminator
critic = Discriminator.Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(critic)

#定义优化器
opt_gen = torch.optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
opt_critic = torch.optim.Adam(critic.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))

#定义随机噪声
fixed_noise = torch.randn(size = (16,Z_DIM,1,1),device=device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

gen.train()
critic.train()

step = 0

for epoch in range(NUM_EPOCHS):
    for batch_idx,(data,_) in enumerate(dataLoader):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        #Train: Critic : max[critic(real)] - E[critic(fake)]
        loss_critic = 0
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(size = (cur_batch_size,Z_DIM,1,1),device=device)
            fake_img = gen(noise)
            #使用reshape主要是将最后的维度从[1,1,1,1]=>[1]
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake_img).reshape(-1)

            gp = gradient_penality(critic,data,fake_img,device=device)

            loss_critic = -(torch.mean(critic_real)- torch.mean(critic_fake)) + LAMBDA_GP*gp
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        #将维度从[1,1,1,1]=>[1]
        gen_fake = critic(fake_img).reshape(-1)
        #max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch[{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataLoader)}\
                Loss D: {loss_critic:.6f},Loss G: {loss_gen:.6f}"
            )
            with torch.no_grad():
                fake_img = gen(fixed_noise)
                DrawGen(gen,epoch,fixed_noise)
                img_grid_real = torchvision.utils.make_grid(
                    data,normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake_img,normalize=True
                )
                writer_real.add_image("RealImg",img_grid_real,global_step=step)
                writer_fake.add_image("fakeImg",img_grid_fake,global_step=step)
            step += 1
            gen.train()
            critic.train()
torch.save(obj=gen,f='models/gen.pth')
torch.save(obj=critic,f='models/disc.pth')
