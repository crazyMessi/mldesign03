import argparse
import sys
import io
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import *
import progressbar
import utils.option as option
from utils.model_controller import *
torch.autograd.set_detect_anomaly(True)

pro = progressbar.ProgressBar()

train_opt = option.get_train_opt()

cuda = True if torch.cuda.is_available() else False
test = True

data_path = train_opt['data_path']
model_name = train_opt['model_name']
# 设置测试lossfun类型(不影响训练)。如果设成none则使用生成器内部的loss
test_loss_func = fixed_loss_G()


if train_opt['if_fitlog']:
    import fitlog
    log_name = 'logs/test'
    os.makedirs(log_name, exist_ok=True)
    fitlog.set_log_dir(log_name)  # 设置log文件夹为'logs/', fitlog在每次运行的时候会默认以时间戳的方式在里面生成新的log
    fitlog.add_hyper(train_opt.get_fitlog_hyper())

transforms_ = []
if train_opt['channels'] == 1:
    transforms_.append(transforms.ToPILImage())
    transforms_.append(transforms.Grayscale(num_output_channels=1))
    transforms_.append(transforms.ToTensor())
    transforms_.append(transforms.Normalize(0.5, 0.5))
else:
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# 修改成本地存放数据集地址
dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_),
                        batch_size=train_opt['bs'], shuffle=True, num_workers=0)
train_opt['dataloader_length'] = len(dataloader)

val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='val'),
                            batch_size=20, shuffle=False, num_workers=0)

test_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='test'),
                             batch_size=10, shuffle=False, num_workers=0)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = model_selector(train_opt)
start_rp = train_opt['epoch']

# 目前只支持手动选择要开始的模型
if train_opt['epoch'] > 0:
    model_root = option.askopenfilename()
    model.load_state_dict(torch.load(model_root))
else:
    # 为网络参数赋初值
    model.apply(weights_init_kaiming)
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
    save_image(img_sample, train_opt.get_img_root() + '/%s.png' % batches_done, nrow=5, normalize=True)


def cal_test_loss():
    imgs = next(iter(test_dataloader))
    real_A = imgs['B'].type(Tensor)
    real_B = imgs['A'].type(Tensor)
    fake_B = model.generator(real_A)
    if test_loss_func:
        test_lose_pixel = test_loss_func(fake_B, real_B).detach()
    else:
        test_lose_pixel = model.g_loss_func(fake_B, real_B).detach()
    return {'test_loss_pixel': test_lose_pixel}


# ----------
#  Training
# ----------
model.train()

min_tloss = 500

bs_count = len(dataloader)
pro.start(train_opt['ep'] * bs_count)

for epoch in range(train_opt['epoch'], train_opt['ep']):
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

    # 计算一个epoch中的指标均值
    result = {}
    for value in loss_dic:
        for key, val in value.items():
            result.setdefault(key, []).append(val)

    result = {i: sum(result[i]) / len(result[i]) for i in result}
    # 计算loss
    result.update(cal_test_loss())
    if train_opt['if_fitlog']:
        fitlog.add_metric(result, epoch)
        fitlog.add_best_metric(result)

    # 每50轮保存模型参数
    if epoch % 50 == 1 or epoch == train_opt['ep'] - 1:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (train_opt.get_model_root(), model_name, epoch))
pro.finish()
code = 1
if test:
    test_path = sys.path[0] + '/test.py'
    com = 'python \"%s\" --model_dir \"%s\"' % (test_path, train_opt.get_model_root())
    com += option.opt_to_str(train_opt.opt)
    code = os.system(com)

if fitlog:
    if code == 0:
        fitlog.finish()