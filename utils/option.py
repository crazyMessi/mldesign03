'''
由opt决定的文件树
'''
import argparse
import os
import tkinter as tk
from tkinter import filedialog
from utils.model_controller import valid_model_name


class Train_opt:
    def __init__(self, opt, root=os.getcwd() + '/output'):
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
        return self.get_root() + '/log'

    # 获得model存放路径
    def get_model_root(self):
        return self.get_root() + '/model'

    # 获得img存放路径
    def get_img_root(self):
        return self.get_root() + '/img'

    # 返回用于命名文件夹的超参
    def get_key_hyper(self):
        k = ['lrG', 'lrD', 'bs', 'ep']
        v = {key: value for key, value in self.opt.items() if key in k}
        return v

    def get_fitlog_hyper(self):
        k = ['lrG', 'lrD', 'bs', 'ep', 'lrG_d', 'lrD_d', 'model_name', 'b1', 'b2', 'weight_pic',
             'g_loss_func', 'channels', 'res_learn', 'dropout']
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
            had_set = False
            for name in valid_model_name:
                if model_dir.find(name) >= 0 & model_dir.find('model') >= 0:
                    had_set = True
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


def askopenfilename():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames()


def get_base_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_mode', type=str, default='null', help='设置模型训练还是测试')
    # 为了找到训练模型参数地址，要与train.py中model_name参数一致
    parser.add_argument('--model_name', type=str, default="null", help='模型名')
    parser.add_argument('--dropout', type=int, default=1, help='是否使用dropout')
    parser.add_argument('--res_learn', type=int, default=1, help='是否使用残差学习')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')

    parser.add_argument('--lrG', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--lrD', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--bs', type=int, default=8, help='size of the batches')
    parser.add_argument('--ep', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--lrG_d', type=int, default=90, help='G lr down')
    parser.add_argument('--lrD_d', type=int, default=10, help='D lr down')
    parser.add_argument('--g_loss_func', type=str, default='L1', help='L1表示L1;fixed_L1表示修正的L1')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--weight_pic', type=float, default=10, help='计算生成器loss时,pic_loss的比例')
    # parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')

    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_height', type=int, default=64, help='size of image height')
    parser.add_argument('--img_width', type=int, default=64, help='size of image width')
    parser.add_argument('--sample_interval', type=int, default=500,
                        help='interval between sampling of images from generators')
    parser.add_argument('--if_remove', type=int, default=1, help='是否需要移除模型')
    parser.add_argument('--data_path', type=str, default='fontdata', help='数据集位置')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='interval between model checkpoints')
    return parser


def get_train_opt():
    parser = get_base_parse()
    parser.add_argument('--if_fitlog', type=int, default=1, help='是否使用fitlog')
    parser.add_argument('--if_test', type=int, default=1, help='是否在执行完备后test')
    opt = parser.parse_args()
    opt = Train_opt(opt)
    opt['model_mode'] = 'train'
    return opt


def get_test_opt():
    parser = get_base_parse()  # 创建解析器对象 可以添加参数
    parser.add_argument('--model_dir', type=str, default="test", help='模型文件夹')
    opt = parser.parse_args()
    opt = Test_opt(opt)
    opt['model_mode'] = 'test'
    return opt