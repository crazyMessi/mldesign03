import torch


class Adam_Optimizer:
    # freq表示学习率折半的频率 每更新freq次参数学习率折半 (对于lrd, freq=lrG_d*dataloader_length) 当freq取0时, 不更新
    def __init__(self, parameters, lr, betas, freq=0):
        super(Adam_Optimizer, self).__init__()
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)
        self.freq = freq
        self.times = 0
        self.lr = lr

    def step(self):
        if self.times % self.freq == 0 and self.times > 0:
            self.lr *= 0.5
            self.optimizer.param_groups[0]['lr'] = self.lr
            print(self.lr)
        self.optimizer.step()
        self.times += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

