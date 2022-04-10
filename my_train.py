import argparse
import sys
import io
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import *
import progressbar
import utils.option as option
from utils.model_controller import *
torch.autograd.set_detect_anomaly(True)

pro = progressbar.ProgressBar()

my_opt = option.get_train_opt()

cuda = True if torch.cuda.is_available() else False

data_path = my_opt['data_path']
model_name = my_opt['model_name']
# 设置测试lossfun类型(不影响训练)。如果设成none则使用生成器内部的loss
test_loss_func = fixed_loss_G()


transforms_ = []
if my_opt['channels'] == 1:
    transforms_.append(transforms.ToPILImage())
    transforms_.append(transforms.Grayscale(num_output_channels=1))
    transforms_.append(transforms.ToTensor())
    transforms_.append(transforms.Normalize(0.5, 0.5))
else:
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# 修改成本地存放数据集地址
dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_),
                        batch_size=my_opt['bs'], shuffle=True, num_workers=0)
my_opt['dataloader_length'] = len(dataloader)

val_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='val'),
                            batch_size=20, shuffle=False, num_workers=0)

# 每次测试整个测试集的得分,否则fitlog会记录最好的那个batch
test_dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='test'),
                             batch_size=80, shuffle=False, num_workers=0)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = model_selector(my_opt)
with io.open(my_opt.get_model_root() + my_opt['model_name'] +'.txt', 'a', encoding='utf-8') as file:
        file.write(str(model))
start_rp = my_opt['epoch']
loss_test = 0
ep_list = []

# 目前只支持手动选择要开始的模型
if my_opt['epoch'] > 0:
    model_root = option.askopenfilename()
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
    save_image(img_sample, my_opt.get_img_root() + '/%s.png' % batches_done, nrow=5, normalize=True)


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


def save_test_imgs(ep):
    t = np.zeros([1,len(test_dataloader)])
    for i, imgs in enumerate(test_dataloader):
            real_A = imgs['B'].type(Tensor)
            real_B = imgs['A'].type(Tensor)
            fake_B = model.generator(real_A)    
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
            save_image(img_sample, '%s/test%s_ep%d.png'%(my_opt.get_img_root(),i,ep), nrow=10, normalize=True)
            if test_loss_func:
                t[0][i] = (test_loss_func(fake_B, real_B)).item()
            else:
                t[0][i] = (model.g_loss_func(fake_B, real_B)).item()
    # if loss_test:
    #     loss_test = np.array(loss_test) + t
    # else:
    #     loss_test = t



# ----------
#  Training
# ----------
model.train()

min_tloss = 500
test_freq = 10

bs_count = len(dataloader)
pro.start(my_opt['ep'] * bs_count)

for epoch in range(my_opt['epoch'], my_opt['ep']):
    loss_dic = []
    for i, batch in enumerate(dataloader):
        # Model inputs
        source = batch['B'].type(Tensor)
        target = batch['A'].type(Tensor)
        if my_opt['use_val']>0:
            val_imgs = next(iter(test_dataloader))
            val_source = val_imgs['B'].type(Tensor)
            model.additional_step(val_source)
        
        # 训练模型
        loss_dic.append(model.step(source, target))
        
        batches_done = epoch * len(dataloader) + i
        # If at sample interval save image
        if int(batches_done * my_opt['bs'] / 8) % int(my_opt['sample_interval']) == 0:
            sample_images(batches_done)
        # 打印进度条
        pro.update(i + epoch * bs_count)

    # 计算一个epoch中的指标均值
    result = {}
    for value in loss_dic:
        for key, val in value.items():
            result.setdefault(key, []).append(val)

    result = {i: sum(result[i]) / len(result[i]) for i in result}
    # 计算test_loss
    result.update(cal_test_loss())
    if my_opt['if_fitlog']:
        if epoch == my_opt['epoch']:
            import fitlog
            log_name = 'logs/test'
            os.makedirs(log_name, exist_ok=True)
            fitlog.set_log_dir(log_name)  # 设置log文件夹为'logs/', fitlog在每次运行的时候会默认以时间戳的方式在里面生成新的log
            fitlog.add_hyper(my_opt.get_fitlog_hyper())


        fitlog.add_metric(result, epoch)
        # 注意这会选取表现最好的一个test batch作为其得分
        fitlog.add_best_metric(result)

    # 每test_freq轮保存模型参数或者测试模型
    if epoch % test_freq == 0 or epoch == my_opt['ep'] - 1:
        if my_opt['if_save'] == 1:
            torch.save(model.state_dict(), '%s/%s_%d.pth' % (my_opt.get_model_root(), model_name, epoch))
        if my_opt['if_test'] == 0:
            save_test_imgs(ep = epoch)
            ep_list.append(epoch)

if my_opt['if_save'] == -1:
    torch.save(model, '%s/%s_%d.pth' % (my_opt.get_model_root(), model_name, epoch))

pro.finish()
code = 0
if my_opt['if_test']>0:
    test_path = sys.path[0] + '/test.py'
    com = 'python \"%s\" --model_dir \"%s\"' % (test_path, my_opt.get_model_root())
    com += option.opt_to_str(my_opt.opt)
    code = os.system(com)


if fitlog:
    if code == 0:
        fitlog.finish()

