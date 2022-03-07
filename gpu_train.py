import argparse
import fitlog
import io

from torchvision.utils import save_image

from torch.utils.data import DataLoader

import numpy as np
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        # ipdb.set_trace()
        rootPath = root + '/{}'.format(mode)
        filename = os.listdir(rootPath)
        path = rootPath + '/' + filename[0]

        self.imgs = np.load(path)

    def __getitem__(self, index):
        img_A_a = self.imgs[index][:, :64, :]
        img_B_b = self.imgs[index][:, 64:, :]

        img_A = self.transform(img_A_a.astype(np.uint8))  # 京黑
        img_B = self.transform(img_B_b.astype(np.uint8))  # 黑体

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.imgs)


class Adam_Optimizer:
    # freq表示学习率折半的频率 每更新freq次参数学习率折半 (对于lrd, freq=lrG_d*dataloader_length) 当freq取0时, 不更新
    def __init__(self, parameters, lr, betas, freq=0):
        super(Adam_Optimizer, self).__init__()
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)
        self.freq = freq
        self.times = 0
        self.lr = lr

    def step(self):
        if self.times % self.freq == 0 and self.times > 0:
            self.lr *= 0.5
            self.optimizer.param_groups[0]['lr'] = self.lr
            print(self.lr)
        self.optimizer.step()
        self.times += 1

    def zero_grad(self):
        self.optimizer.zero_grad()


from torch import nn

generator_loss_fun = torch.nn.L1Loss()
discriminator_loss_fun = torch.nn.MSELoss()


# ===================================
#              网络单元
# ===================================
class Encoder(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4):

        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=ks, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, normalize=True):
        super(Decoder, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4):

        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=ks, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, if_crop=True):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.if_crop = if_crop

    def forward(self, x, skip_input):
        # ipdb.set_trace()
        x = self.model(x)
        if self.if_crop > 0:
            x = torch.cat((x, skip_input), 1)
        else:
            x = torch.cat((x, x), 1)
        return x


# ===================================
#              子模型
# ===================================
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, if_crop=True, loss_fun=generator_loss_fun):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)  # 不需要正规化了

        self.up1 = UNetUp(512, 512, dropout=0.5, if_crop=if_crop)
        self.up2 = UNetUp(1024, 512, dropout=0.5, if_crop=if_crop)
        self.up3 = UNetUp(1024, 256, if_crop=if_crop)
        self.up4 = UNetUp(512, 128, if_crop=if_crop)
        self.up5 = UNetUp(256, 64, if_crop=if_crop)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )
        self.loss_fun = loss_fun

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)  # x:[1, 3, 64, 64]  d1:[1, 64, 32, 32]
        d2 = self.down2(d1)  # d2:[1,128,16,16]
        d3 = self.down3(d2)  # d3:[1,256,8,8]
        d4 = self.down4(d3)  # d4:[1,512,4,4]
        d5 = self.down5(d4)  # d5:[1,512,2,2]
        d6 = self.down6(d5)  # d6:[1,512,1,1]

        u1 = self.up1(d6, d5)  # u1:[1,1024,2,2]
        u2 = self.up2(u1, d4)  # u2:[1,1024,4,4]
        u3 = self.up3(u2, d3)  # u3:[1,1024,8,8]
        u4 = self.up4(u3, d2)  # u4:[1,1024,16,16]
        u5 = self.up5(u4, d1)  # u5:[1,512,32,32]
        return self.final(u5)

    def loss(self, generate, target):
        return self.loss_fun(generate, target)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, loss_fun=discriminator_loss_fun):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        self.loss_fun = loss_fun

    def forward(self, img_A, img_B):
        # Concatenate image and condition image
        # by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

    def loss(self, pred, real):
        return self.loss_fun(pred, real)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, loss_fun=generator_loss_fun):
        super(AutoEncoder, self).__init__()
        self.down1 = Encoder(in_channels, 64, normalize=False)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512, dropout=0.5)
        self.down5 = Encoder(512, 512, dropout=0.5)
        self.down6 = Encoder(512, 512, dropout=0.5, normalize=False)  # 不需要正规化了

        self.up1 = Decoder(512, 1024, dropout=0.5)
        self.up2 = Decoder(1024, 1024, dropout=0.5)
        self.up3 = Decoder(1024, 512)
        self.up4 = Decoder(512, 256)
        self.up5 = Decoder(256, 128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )
        self.loss_fun = loss_fun

    def forward(self, x):
        # 上采样
        d1 = self.down1(x)  # x:[1, 3, 64, 64]  d1:[1, 64, 32, 32]
        d2 = self.down2(d1)  # d2:[1,128,16,16]
        d3 = self.down3(d2)  # d3:[1,256,8,8]
        d4 = self.down4(d3)  # d4:[1,512,4,4]
        d5 = self.down5(d4)  # d5:[1,512,2,2]
        d6 = self.down6(d5)  # d6:[1,512,1,1]
        # 下采样
        u1 = self.up1(d6)  # u1:[1,1024,2,2]
        u2 = self.up2(u1)  # u2:[1,1024,4,2]
        u3 = self.up3(u2)  # u3:[1,1024,8,2]
        u4 = self.up4(u3)  # u4:[1,1024,16,16]
        u5 = self.up5(u4)  # u5:[1,512,32,32]
        x = self.final(u5)
        return x

    def loss(self, x, y):
        return self.loss_fun(x, y)


# ===================================
#              顶级模型
# ===================================
class GAN(nn.Module):
    def __init__(self, train_opt=None, generator=GeneratorUNet(), discriminator=Discriminator()):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        if isinstance(train_opt, dict):
            self.optimizer_G = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrG'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrG_d'] * train_opt['dataloader_length'])

            self.optimizer_D = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrD'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrD_d'] * train_opt['dataloader_length'])
            self.train_opt = train_opt

    def forward(self, source, target):
        generate = self.generator(source)
        source_generate = self.discriminator(generate, source)
        source_target = self.discriminator(target, source)
        source_generate2 = self.discriminator(generate.detach(), source)
        return generate, source_generate, source_target, source_generate2

    def loss(self, generate, target, source_generate, source_target, source_generate2):
        invalid = source_target.clone().detach() * 0
        valid = invalid + 1
        # 计算生成模型误差
        loss_pixel = self.generator.loss(generate, target)
        loss_sVg = self.discriminator.loss(source_generate, valid)
        loss_G = loss_sVg + self.train_opt['weight_pic'] * loss_pixel
        # 分辨源图像和目标图像
        loss_real = self.discriminator.loss(source_target, valid)
        # 分辨源图像和生成图像
        loss_fake = self.discriminator.loss(source_generate2, invalid)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_G, loss_D, loss_sVg, loss_pixel

    def step(self, source, target):
        generate, source_generate, source_target, source_generate2 = self(source, target)
        loss_G, loss_D, loss_sVg, loss_pixel = self.loss(generate, target, source_generate,
                                                         source_target, source_generate2)
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()
        loss_dic = {'loss_G': loss_G.item(), 'loss_D': loss_D.item(), 'loss_pixel': loss_pixel.item(),
                    'loss_sVg': loss_sVg.item()}
        return loss_dic


# 基于AutoEncoder的图片生成器
class AutoEncoderGen(nn.Module):
    def __init__(self, train_opt=None, generator=AutoEncoder()):
        super(AutoEncoderGen, self).__init__()
        self.generator = generator
        if isinstance(train_opt, dict):
            self.optimizer_G = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrG'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrG_d'] * train_opt['dataloader_length'])

    def loss(self, x, y):
        return self.generator.loss_fun(x, y)

    def forward(self, x):
        return self.generator(x)

    # 只有作为顶级模型时该方法有效合法
    def step(self, x, y):
        generate = self(x)
        loss_pixel = self.loss(y, generate) * 10
        self.optimizer_G.zero_grad()
        loss_pixel.backward()
        self.optimizer_G.step()
        loss_dic = {'loss_pixel': loss_pixel.item()}
        return loss_dic


import os
import tkinter as tk
from tkinter import filedialog

valid_model_name = ['GAN', 'AutoEncoderGen', 'pic2pic']


class Train_opt:
    def __init__(self, opt, root='/output'):
        super(Train_opt, self).__init__()
        # 将opt转为字典类型
        if not isinstance(opt, dict):
            self.opt = vars(opt)
        else:
            self.opt = opt
        self.root = root
        self.mk_use_dirs()

    def __getitem__(self, item):
        return self.opt.__getitem__(item)

    def __setitem__(self, key, value):
        self.opt.__setitem__(key, value)

    # 获得本次输出的根目录
    def get_root(self):
        # 格式：模型/G学习率_D学习率_批大小_epoch
        opt = self.opt
        dir_name = ''
        k_hyper = self.get_key_hyper()
        for k, v in k_hyper.items():
            dir_name += str(k) + str(v)
        root = '%s/%s/%s/train' % (self.root, opt['model_name'], dir_name)
        return root

    # 获得log存放路径
    def get_log_root(self):
        return self.get_root() + '/log/'

    # 获得model存放路径
    def get_model_root(self):
        return self.get_root() + '/model/'

    # 获得img存放路径
    def get_img_root(self):
        return self.get_root() + '/img/'

    # 返回用于命名文件夹的超参
    def get_key_hyper(self):
        k = ['lrG', 'lrD', 'bs', 'ep']
        v = {key: value for key, value in self.opt.items() if key in k}
        return v

    def get_fitlog_hyper(self):
        k = ['lrG', 'lrD', 'bs', 'ep', 'model_name']
        v = {key: value for key, value in self.opt.items() if key in k}
        return v

    # 命名可能需要的文件夹
    def mk_use_dirs(self):
        print('创建 ' + self.get_img_root())
        print('创建 ' + self.get_log_root())
        print('创建 ' + self.get_model_root())
        os.makedirs(self.get_log_root(), exist_ok=True)
        os.makedirs(self.get_img_root(), exist_ok=True)
        os.makedirs(self.get_model_root(), exist_ok=True)


class Test_opt:
    def __init__(self, opt):
        super(Test_opt, self).__init__()
        # 将opt转为字典类型
        if not isinstance(opt, dict):
            self.opt = vars(opt)
        else:
            self.opt = opt

        try:
            model_dir = self.opt['model_dir']
            had_set = model_dir.split('/')[-4] not in valid_model_name
        except (KeyError, IndexError):
            print("未指定合法目录,请手动选择待测试模型位置")
            had_set = False

        while self.opt['model_name'] not in valid_model_name or not had_set:
            root = tk.Tk()
            root.withdraw()
            model_dir = filedialog.askdirectory()
            self.opt['model_name'] = model_dir.split('/')[-4]
            had_set = True

        self.mode_dir = model_dir
        self.test_out = model_dir.replace('train', 'test', 1)
        self.test_out = self.test_out.replace('/model', '')
        self.mk_use_dirs()

    def __getitem__(self, item):
        return self.opt.__getitem__(item)

    def __setitem__(self, key, value):
        self.opt.__setitem__(key, value)

    def get_root(self):
        return self.test_out

    def get_img_root(self):
        return self.get_root() + '/test_img'

    def get_log_root(self):
        return self.get_root() + '/test_log'

    def get_model_root(self):
        return self.mode_dir

    def mk_use_dirs(self):
        print('创建 ' + self.get_img_root())
        print('创建 ' + self.get_log_root())
        os.makedirs(self.get_log_root(), exist_ok=True)
        os.makedirs(self.get_img_root(), exist_ok=True)


valid_model_name = ['GAN', 'AutoEncoderGen', 'pic2pic']


# 为网络参数赋正态分布的初值
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 根据opt选取模型
def model_selector(opt):
    if isinstance(opt, dict):
        model_name = opt['model_name']
    else:
        model_name = opt

    generator_list = {'UNet': GeneratorUNet(), 'GAN': GeneratorUNet(if_crop=False), 'AutoEncoder': AutoEncoder()}
    discriminator_list = {'Discriminator': Discriminator()}

    while model_name not in valid_model_name:
        print('未输入正确模型名 请输入正确模型名\n')
        print(valid_model_name)
        model_name = input()

    if model_name == 'AutoEncoderGen':
        model = AutoEncoderGen(train_opt=opt, generator=generator_list['AutoEncoder'])

    if model_name == 'GAN':
        generator = generator_list['GAN']
        discriminator = discriminator_list['Discriminator']
        model = GAN(train_opt=opt, generator=generator, discriminator=discriminator)

    if model_name == 'pic2pic':
        generator = generator_list['UNet']
        discriminator = discriminator_list['Discriminator']
        model = GAN(train_opt=opt, generator=generator, discriminator=discriminator)

    return model


valid_model_name = ['GAN', 'AutoEncoderGen', 'pic2pic']


# 为网络参数赋正态分布的初值
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 根据opt选取模型
def model_selector(opt):
    if isinstance(opt, dict):
        model_name = opt['model_name']
    else:
        model_name = opt

    generator_list = {'UNet': GeneratorUNet(), 'GAN': GeneratorUNet(if_crop=False), 'AutoEncoder': AutoEncoder()}
    discriminator_list = {'Discriminator': Discriminator()}

    while model_name not in valid_model_name:
        print('未输入正确模型名 请输入正确模型名\n')
        print(valid_model_name)
        model_name = input()

    if model_name == 'AutoEncoderGen':
        model = AutoEncoderGen(train_opt=opt, generator=generator_list['AutoEncoder'])

    if model_name == 'GAN':
        generator = generator_list['GAN']
        discriminator = discriminator_list['Discriminator']
        model = GAN(train_opt=opt, generator=generator, discriminator=discriminator)

    if model_name == 'pic2pic':
        generator = generator_list['UNet']
        discriminator = discriminator_list['Discriminator']
        model = GAN(train_opt=opt, generator=generator, discriminator=discriminator)

    return model


try:
    import ipdb
except:
    import pdb as ipdb

import time
import progressbar

pro = progressbar.ProgressBar()

parser = argparse.ArgumentParser()  # 创建解析器对象 可以添加参数
parser.add_argument('--model_name', type=str, default='test', help='模型名称')
parser.add_argument('--lrG', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--lrD', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--bs', type=int, default=8, help='size of the batches')
parser.add_argument('--ep', type=int, default=200, help='number of epochs of training')
parser.add_argument('--lrG_d', type=int, default=90, help='G lr down')
parser.add_argument('--lrD_d', type=int, default=10, help='D lr down')
parser.add_argument('--weight_pic', type=float, default=10, help='计算生成器loss时,pic_loss的比例')
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

opt = parser.parse_args()
if_fitlog = True

data_path = '../input/mldesign03/fontdata'
TPU = True
test = True

train_opt = Train_opt(opt)
# Initialize generator and discriminator
model_name = train_opt['model_name']

if if_fitlog:
    fitlog.set_log_dir('logs/')  # 设置log文件夹为'logs/', fitlog在每次运行的时候会默认以时间戳的方式在里面生成新的log
    fitlog.add_hyper(train_opt.get_fitlog_hyper())

transforms_ = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# 修改成本地存放数据集地址
dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_),
                        batch_size=opt.bs, shuffle=True, num_workers=0)
train_opt['dataloader_length'] = len(dataloader)

val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='train'),
                            batch_size=20, shuffle=False, num_workers=0)

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
    dev = torch.cuda.device


model = model_selector(train_opt.opt)
# 为网络参数赋初值
model.apply(weights_init_normal)


if TPU:
    model.cuda()


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs['B'].type(Tensor)
    real_B = imgs['A'].type(Tensor)
    fake_B = model.generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    # ipdb.set_trace()
    save_image(img_sample, train_opt.get_img_root() + '%s.png' % batches_done, nrow=5, normalize=True)


# ----------
#  Training
# ----------
model.train()

min_tloss = 500
tloss_res = {}

bs_count = len(dataloader)
pro.start(train_opt['ep'] * bs_count)

for epoch in range(opt.epoch, opt.ep):

    for i, batch in enumerate(dataloader):

        # Model inputs
        source = batch['B'].type(Tensor)
        target = batch['A'].type(Tensor)
        loss_dic = model.step(source, target)

        batches_done = epoch * len(dataloader) + i
        # If at sample interval save image
        if int(batches_done * train_opt['bs'] / 8) % int(train_opt['sample_interval']) == 0:
            sample_images(batches_done)
        # 打印进度条
        pro.update(i + epoch * bs_count)

    avg_loss = 0
    tloss_res[epoch] = avg_loss

    if if_fitlog:
        fitlog.add_metric(loss_dic, epoch)
        fitlog.add_best_metric(loss_dic)

    # 每50轮保存模型参数
    if epoch > 0 and (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (train_opt.get_model_root(), model_name, epoch))
        # torch.save(discriminator.state_dict(),
        #            train_opt.get_model_root() + '/discriminator_%d.pth' % epoch)
        # torch.save(model.state_dict())
    # 保存loss最小时的模型参数
    # if tloss_res[epoch] < min_tloss:
    #     min_tloss = tloss_res[epoch]
    #     tloss_res['min'] = tloss_res[epoch]
    #     tloss_res['minepoch'] = epoch
    #     torch.save(model.state_dict(), '%s/%s_min.pth' % (train_opt.get_model_root(), model_name))

with io.open(train_opt.get_log_root() + 'list_loss.txt', 'a', encoding='utf-8') as file:
    file.write('tloss_res: {} \n'.format(tloss_res))
pro.finish()
if test:
    os.system('python test.py --model_dir \"%s\" --model_name %s' % (train_opt.get_model_root(), model_name))
