import matplotlib.pyplot as plt
import numpy as np
from scipy import rand
import torch
import sys
sys.path.append('E:\ZHUODONG LI\Work\AI\mldesign03')
from utils.model_controller import fixed_loss_G
def show_tensor(ts, i=0):
    ts = torch.mean(ts,1)
    array_img = ts.cpu().numpy()
    plt.imshow(array_img[i])

pass

lo = fixed_loss_G()
x = torch.rand(80,1,64,64)
y = torch.rand(80,1,64,64)
l = 0
for i in range(len(x)):
    l += lo(x[i],y[i])
ll = lo(x,y)
pass