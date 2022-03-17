import argparse
import sys
from tkinter import filedialog

import fitlog
import io

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from dataset import *
import utils.file_manager as fm

from utils.model_controller import *
from PIL import Image

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
# 有的地方(比如kaggle)目录比较奇怪
parser.add_argument('--data_path', type=str, default='fontdata', help='数据集位置')

opt = parser.parse_args()
if_fitlog = True

cuda = True if torch.cuda.is_available() else False
test = True

train_opt = fm.Train_opt(opt)
data_path = train_opt['data_path']
# Initialize generator and discriminator
model_name = train_opt['model_name']

if if_fitlog:
    log_name = 'logs/'
    # for _, v in train_opt.get_fitlog_hyper().items():
    #     log_name = log_name + str(v) + '_'
    os.makedirs(log_name, exist_ok=True)
    fitlog.set_log_dir(log_name)  # 设置log文件夹为'logs/', fitlog在每次运行的时候会默认以时间戳的方式在里面生成新的log
    fitlog.add_hyper(train_opt.get_fitlog_hyper())

transforms_ = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# 修改成本地存放数据集地址
dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_),
                        batch_size=opt.bs, shuffle=True, num_workers=0)
train_opt['dataloader_length'] = len(dataloader)

val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='train'),
                            batch_size=20, shuffle=False, num_workers=0)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = model_selector(train_opt.opt)
start_rp = train_opt['epoch']

# 目前只支持手动选择要开始的模型
if train_opt['epoch'] > 0:
    model_root = fm.askopenfilename()
    model.load_state_dict(torch.load(model_root))
else:
    # 为网络参数赋初值
    model.apply(weights_init_normal)
if cuda:
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

    loss_dic = []
    for i, batch in enumerate(dataloader):

        # Model inputs
        source = batch['B'].type(Tensor)
        target = batch['A'].type(Tensor)
        loss_dic.append(model.step(source, target))
        batches_done = epoch * len(dataloader) + i
        # If at sample interval save image
        if int(batches_done * train_opt['bs'] / 8) % int(train_opt['sample_interval']) == 0:
            sample_images(batches_done)
        # 打印进度条
        pro.update(i + epoch * bs_count)

    avg_loss = 0
    tloss_res[epoch] = avg_loss

    # 计算一个epoch中的指标均值
    result = {}
    for value in loss_dic:
        for key, val in value.items():
            result.setdefault(key, []).append(val)

    result = {i: sum(result[i]) / len(result[i]) for i in result}
    if if_fitlog:
        fitlog.add_metric(result, epoch)
        fitlog.add_best_metric(result)

    # 每50轮保存模型参数
    if epoch > 0 and (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (train_opt.get_model_root(), model_name, epoch))

with io.open(train_opt.get_log_root() + 'list_loss.txt', 'a', encoding='utf-8') as file:
    file.write('tloss_res: {} \n'.format(tloss_res))
pro.finish()
if test:
    test_path = sys.path[0]+'/test.py'
    os.system('python \"%s\" --model_dir \"%s\" --model_name %s --data_path \"%s\"'
              % (test_path, train_opt.get_model_root(), model_name, data_path))
