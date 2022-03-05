'''
由opt决定的文件树
'''
import os
import tkinter as tk
from tkinter import filedialog
from utils.tools import valid_model_name


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

    def get_root(self):
        return self.test_out

    def get_img_root(self):
        return self.get_root() + '/test_img'

    def get_log_root(self):
        return self.get_root() + '/test_log'

    def get_model_root(self):
        return self.mode_dir

    def mk_use_dirs(self):
        print('创建 '+self.get_img_root())
        print('创建 '+self.get_log_root())
        os.makedirs(self.get_log_root(), exist_ok=True)
        os.makedirs(self.get_img_root(), exist_ok=True)

