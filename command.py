import os

ep = 402
lrGs = [0.0001, 0.0002]
bss = [8, 16]


def train(name):
    for lrG in lrGs:
        for bs in bss:
            os.system('python my_train.py --model_name %s --ep %d --lrG %f --bs %d' % (name, ep, lrG, bs))


model_name = 'pic2pic'
train(model_name)

model_name = 'GAN'
train(model_name)
