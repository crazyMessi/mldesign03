import torch
from tensorboardX import SummaryWriter
from myModel import *

model = AutoEncoderGen()
source = torch.rand(8, 3, 64, 64)
target = torch.rand(8, 3, 64, 64)
with SummaryWriter(comment='GAN') as w:
    w.add_graph(model, source)