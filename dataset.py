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
