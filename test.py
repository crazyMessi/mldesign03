import argparse
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.file_manager import *
from dataset import *
import torch

from utils.model_controller import model_selector

try:
    import ipdb
except:
    import pdb as ipdb

parser = argparse.ArgumentParser()  # 创建解析器对象 可以添加参数
# 为了找到训练模型参数地址，要与train.py中model_name参数一致
parser.add_argument('--model_dir', type=str, default="test", help='模型文件夹')
parser.add_argument('--model_name', type=str, default="test", help='模型名')
opt = parser.parse_args()
my_opt = Test_opt(opt)
cuda = True if torch.cuda.is_available() else False
data_path = os.getcwd()+"/fontdata/"
model = model_selector(my_opt.opt['model_name'])

if cuda:
    model.cuda()

transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# ImageDataset第一个参数改成个人数据集存放地址
val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='test'),
                            batch_size=20, shuffle=False, num_workers=0)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model_dict = my_opt.get_model_root()
filename = os.listdir(model_dict)

for j in range(len(filename)):
    if os.path.splitext(filename[j])[1] == '.pth':
        model_location = '%s/%s' % (model_dict, filename[j])
        model.load_state_dict(torch.load(model_location))
        model.eval()
        for i, batch in enumerate(val_dataloader):
            print(i)
            real_A = Variable(batch['B'].type(Tensor))
            real_B = Variable(batch['A'].type(Tensor))
            fake_B = model.generator(real_A)
            # 图片存放处
            save_image(fake_B, my_opt.get_img_root()+'/%s.png' % ('img'+str(i) + '_' + filename[j].split('.')[0]), nrow=10,
                       normalize=True)
            save_image(real_B, my_opt.get_img_root()+'/%s.png' % str(i), nrow=10, normalize=True)
    os.remove(model_location)
