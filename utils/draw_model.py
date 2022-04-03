import sys
sys.path.append('E:\ZHUODONG LI\Work\AI\mldesign03')
import torch
from tensorboardX import SummaryWriter
from myModel import *

model = UResGen(3)
source = torch.rand(8, 3, 64, 64)
target = torch.rand(8, 3, 64, 64)
with SummaryWriter(comment='UResGen') as w:
    w.add_graph(model, source)
