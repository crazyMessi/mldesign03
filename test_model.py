import argparse
import re

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn



import glob
import random
import os
import matplotlib
from matplotlib.pyplot import figure
import numpy as np
import torch
import torchvision.utils

from torch.utils.data import Dataset
from PIL import Image
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


class fixed_loss_G(nn.Module):
    def __init__(self):
        super(fixed_loss_G, self).__init__()
        self.loss_G = torch.nn.L1Loss()
        return

    def forward(self, x, y):
        loss_G = self.loss_G(1 - x, 1 - y)
        return loss_G / (torch.mean(1 - y))

# 修正的生成器loss

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',type = str,default='null')
parser.add_argument('--channels',type = int,default=1)
my_opt = parser.parse_args()



MSE = torch.nn.MSELoss()
# 设置损失函数(仅用于查看) 设为none则使用生成器的
view_g_loss_func = fixed_loss_G()

cuda = True if torch.cuda.is_available() else False

transforms_ = []
if my_opt.channels == 1:
    transforms_.append(transforms.ToPILImage())
    transforms_.append(transforms.Grayscale(num_output_channels=1))
    transforms_.append(transforms.ToTensor())
    transforms_.append(transforms.Normalize(0.5, 0.5))
else:
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# ImageDataset第一个参数改成个人数据集存放地址
val_dataloader = DataLoader(ImageDataset('fontdata', transforms_=transforms_, mode='test'),
                            batch_size=20, shuffle=False, num_workers=0)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model_dict = my_opt.model_dir
filename = os.listdir(model_dict)
loss_test = np.zeros([len(filename), len(val_dataloader)])
ep = []

for j in range(len(filename)):
    if os.path.splitext(filename[j])[1] == '.pth':
        model_location = '%s/%s' % (model_dict, filename[j])
        # model.load_state_dict(torch.load(model_location))
        model = torch.load(model_location)
        for i, batch in enumerate(val_dataloader):
            source = Variable(batch['B'].type(Tensor))
            target = Variable(batch['A'].type(Tensor))
            fake_B = model(source)
            # 图片存放处
            save_image(torch.cat((source, target, fake_B), -2), 'test'
                       + '/%s.png' % ('img' + str(i) + '_' + filename[j].split('.')[0]), nrow=10,
                       normalize=True)
            # save_image(target, my_opt.get_img_root() + '/%s.png' % str(i), nrow=10, normalize=True)
            if view_g_loss_func:
                loss_test[j][i] = (view_g_loss_func(fake_B, target)).item()
            else:
                loss_test[j][i] = (model.g_loss_func(fake_B, target)).item()

        # 获得ep数 不稳定
        ep_count = re.findall(r"_\d*.", filename[j])
        ep_count = re.findall(r"\d+", ep_count[-1])
        ep.append(ep_count[-1])

ax = plt.subplot()
step = max(1, int(len(loss_test) / 5))
for i in range(0, len(loss_test), step):
    ax.plot(loss_test[i, :], label=filename[i])
ax.legend()
plt.savefig('test' + '/loss_summary1.png')
plt.figure()
plt.plot(ep, np.mean(loss_test, 1))
plt.savefig('test' + '/loss_summary2.png')
print("测试完成")
