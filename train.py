import argparse
import os

import fitlog
import numpy as np

import time
import datetime
import sys
import io

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
from model import *
from dataset import *
from myModel import AutoEncoder
import utils.file_manager as fm

import torch

try:
    import ipdb
except:
    import pdb as ipdb

parser = argparse.ArgumentParser()  # 创建解析器对象 可以添加参数
parser.add_argument('--model_name', type=str, default='test', help='模型名称')
parser.add_argument('--lrG', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--lrD', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--bs', type=int, default=8, help='size of the batches')
parser.add_argument('--ep', type=int, default=200, help='number of epochs of training')
parser.add_argument('--lrgd', type=int, default=90, help='G lr down')
parser.add_argument('--lrdd', type=int, default=10, help='D lr down')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--dataset_name', type=str, default='test', help='name of the dataset')

parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=64, help='size of image height')
parser.add_argument('--img_width', type=int, default=64, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=20, help='interval between model checkpoints')
parser.add_argument('--if_crop', type=int, default=1, help='上采样时是否使用下采样信息补充')

opt = parser.parse_args()
my_opt = fm.train_opt(opt)
if_fitlog = True
if if_fitlog:
    fitlog.set_log_dir('logs/')  # 设置log文件夹为'logs/', fitlog在每次运行的时候会默认以时间戳的方式在里面生成新的log
    fitlog.add_hyper(my_opt.get_key_hyper())
data_path = os.getcwd() + "/fontdata/"

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 10

cuda = True if torch.cuda.is_available() else False
test = True

# Loss functions
criterion_GAN = torch.nn.MSELoss()  # 均方损失函数
criterion_pixelwise = torch.nn.L1Loss()  # 创建一个衡量输入x(模型预测输出)和目标y之间差的绝对值的平均值的标准

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
# generator = GeneratorUNet()
generator = AutoEncoder()
# generator = GeneratorAutoEncoder()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# ipdb.set_trace()

# 为网络参数赋初值
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
lrG = opt.lrG
lrD = opt.lrD

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lrD, betas=(opt.b1, opt.b2))

# Configure dataloaders

transforms_ = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# 修改成本地存放数据集地址
dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_),
                        batch_size=opt.bs, shuffle=True, num_workers=0)

val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='train'),
                            batch_size=20, shuffle=False, num_workers=0)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs['B'].type(Tensor)
    real_B = imgs['A'].type(Tensor)
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    # ipdb.set_trace()
    save_image(img_sample, my_opt.get_img_root() + '%s.png' % batches_done, nrow=5, normalize=True)


def loss_val():
    batch = next(iter(val_dataloader))
    real_A = batch['B'].type(Tensor)
    real_B = batch['A'].type(Tensor)
    valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
    fake_B = generator(real_A)
    pred_fake = discriminator(fake_B, real_A)
    loss_GAN = criterion_GAN(pred_fake, valid)
    # Pixel-wise loss
    loss_pixel = criterion_pixelwise(fake_B, real_B)

    return loss_pixel.item()


# ----------
#  Training
# ----------
generator.train()
discriminator.train()
prev_time = time.time()


# 设置衰减论
def lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter):
    if epoch % lr_decay_iter:
        return init_lr
    lr = init_lr * 0.5
    optimizer.param_groups[0]['lr'] = lr
    return lr


min_tloss = 500
tloss_res = {}

for epoch in range(opt.epoch, opt.ep):

    ch_lr_avg_loss_depart = []
    ch_lr_avg_loss = 0

    if epoch > 0:
        lrG = lr_scheduler(optimizer_G, lrG, epoch + 1, opt.lrgd)
        # lrD = lr_scheduler(optimizer_D, lrD,epoch+1, opt.lrdd)

    print(lrG)
    print(lrD)

    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = batch['B'].type(Tensor)
        real_B = batch['A'].type(Tensor)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        # GAN loss
        fake_B = generator(real_A)
        # ipdb.set_trace()
        pred_fake = discriminator(fake_B, real_A)
        # ipdb.set_trace()
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        ch_lr_avg_loss_depart.append(loss_G.data.item())

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()
        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.ep * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [ pixel: %f, loss_GAN: %f] ETA: %s" %
              (epoch, opt.ep,
               i, len(dataloader),
               loss_D.item(),
               lambda_pixel * loss_pixel.item(), loss_GAN.item(),
               time_left))

        with io.open(my_opt.get_log_root() + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write('[Epoch: {}] [loss_D: {:.4f}] [loss_pixel: {:.4f}] [loss_GAN: {:.4f}] [Batch: {}/{}] \n'
                       .format(epoch, loss_D.item(), lambda_pixel * loss_pixel.item(), loss_GAN.item(), i,
                               len(dataloader)))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # 计算平均loss和时间
    ch_lr_avg_loss = sum(ch_lr_avg_loss_depart) / len(ch_lr_avg_loss_depart)

    print('----------------------------------------------------------- \n')
    print('avg_loss: {:.4f} \n'.format(ch_lr_avg_loss))

    with io.open(my_opt.get_log_root() + 'loss_time.txt', 'a', encoding='utf-8') as file:
        file.write('[avg_loss: {:.4f}] \n'.format(ch_lr_avg_loss))

    avg_loss = 0
    avg_loss = loss_val()
    tloss_res[epoch] = avg_loss

    if if_fitlog:
        loss_dic = {'loss_D': loss_D.item(), 'loss_GAN': loss_GAN.item(),
                    'lose_pixel': loss_pixel.item() * lambda_pixel, 'avg_loss': avg_loss}
        fitlog.add_metric(loss_dic, epoch)

    # 每50轮保存模型参数
    if epoch > 0 and (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), my_opt.get_model_root() + '/generator_%d.pth' % epoch)
        torch.save(discriminator.state_dict(),
                   my_opt.get_model_root() + '/discriminator_%d.pth' % epoch)
    # 保存loss最小时的模型参数
    if tloss_res[epoch] < min_tloss:
        min_tloss = tloss_res[epoch]
        tloss_res['min'] = tloss_res[epoch]
        tloss_res['minepoch'] = epoch
        torch.save(generator.state_dict(), my_opt.get_model_root() + '/generator_min.pth')
        torch.save(discriminator.state_dict(), my_opt.get_model_root() + '/discriminator_min.pth')

with io.open(my_opt.get_log_root() + 'list_loss.txt', 'a', encoding='utf-8') as file:
    file.write('tloss_res: {} \n'.format(tloss_res))

if test:
    os.system('python test.py --model_dir \"' + my_opt.get_model_root() + '\"')
