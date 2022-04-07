import matplotlib.pyplot as plt
import numpy as np
import torch
def show_tensor(ts, i=0):
    ts = torch.mean(ts,1)
    array_img = ts.cpu().numpy()
    plt.imshow(array_img[i])

pass