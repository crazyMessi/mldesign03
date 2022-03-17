import argparse

import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import myModel
from dataset import *
from utils.file_manager import *
from utils.model_controller import model_selector

parser = argparse.ArgumentParser()  # 创建解析器对象 可以添加参数
# 为了找到训练模型参数地址，要与train.py中model_name参数一致
parser.add_argument('--model_dir', type=str, default="test", help='模型文件夹')
parser.add_argument('--model_name', type=str, default="test", help='模型名')
parser.add_argument('--if_remove', type=str, default=1, help='是否需要移除模型')
parser.add_argument('--data_path', type=str, default='fontdata', help='数据集位置')
opt = parser.parse_args()
my_opt = Test_opt(opt)
cuda = True if torch.cuda.is_available() else False
data_path = my_opt['data_path']
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
loss_test = np.zeros([len(filename), len(val_dataloader)])

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
            save_image(fake_B, my_opt.get_img_root()+'/%s.png' % ('img'+str(i) + '_' + filename[j].split('.')[0]), nrow=10,
                       normalize=True)
            save_image(target, my_opt.get_img_root() + '/%s.png' % str(i), nrow=10, normalize=True)
            loss_test[j][i] = (myModel.generator_loss_fun(fake_B, target)).item()
    if my_opt['if_remove'] > 0:
        os.remove(model_location)

ax = plt.subplot()
step = max(1, int(len(loss_test)/5))
for i in range(0, len(loss_test), step):
    ax.plot(loss_test[i, :], label=filename[i])
ax.legend()
plt.savefig(my_opt.get_log_root()+'/loss.png')
print("测试完成")