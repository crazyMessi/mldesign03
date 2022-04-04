import argparse
import re

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import myModel
from dataset import *
from utils.option import *
from utils.model_controller import *

# 设置损失函数(仅用于查看) 设为none则使用生成器的
view_g_loss_func = fixed_loss_G()

# 获取测试参数
my_opt = get_test_opt()
cuda = True if torch.cuda.is_available() else False
data_path = my_opt['data_path']
model = model_selector(my_opt)

if cuda:
    model.cuda()

transforms_ = []
if my_opt['channels'] == 1:
    transforms_.append(transforms.ToPILImage())
    transforms_.append(transforms.Grayscale(num_output_channels=1))
    transforms_.append(transforms.ToTensor())
    transforms_.append(transforms.Normalize(0.5, 0.5))
else:
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# ImageDataset第一个参数改成个人数据集存放地址
val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='test'),
                            batch_size=20, shuffle=False, num_workers=0)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model_dict = my_opt.get_model_root()
filename = os.listdir(model_dict)
loss_test = np.zeros([len(filename), len(val_dataloader)])
ep = []

for j in range(len(filename)):
    if os.path.splitext(filename[j])[1] == '.pth':
        model_location = '%s/%s' % (model_dict, filename[j])
        model.load_state_dict(torch.load(model_location))
        model.eval()
        for i, batch in enumerate(val_dataloader):
            source = Variable(batch['B'].type(Tensor))
            target = Variable(batch['A'].type(Tensor))
            fake_B = model.generator(source)
            # 图片存放处
            save_image(torch.cat((source, target, fake_B), -2), my_opt.get_img_root()
                       + '/%s.png' % ('img' + str(i) + '_' + filename[j].split('.')[0]), nrow=10,
                       normalize=True)
            # save_image(target, my_opt.get_img_root() + '/%s.png' % str(i), nrow=10, normalize=True)
            if view_g_loss_func:
                loss_test[j][i] = (view_g_loss_func(fake_B, target)).item()
            else:
                loss_test[j][i] = (model.g_loss_func(fake_B, target)).item()

        if my_opt['if_remove'] > 0:
            os.remove(model_location)
        # 获得ep数 不稳定
        ep_count = re.findall(r"_\d*.", filename[j])
        ep_count = re.findall(r"\d+", ep_count[-1])
        ep.append(ep_count[-1])

ax = plt.subplot()
step = max(1, int(len(loss_test) / 5))
for i in range(0, len(loss_test), step):
    ax.plot(loss_test[i, :], label=filename[i])
ax.legend()
plt.savefig(my_opt.get_img_root() + '/loss_summary1.png')
plt.figure()
plt.plot(ep, np.mean(loss_test, 1))
plt.savefig(my_opt.get_img_root() + '/loss_summary2.png')
print("测试完成")
