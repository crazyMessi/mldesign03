import os

ep = 200
lrGs = [0.01, 0.05]
bss = [8,16,40,200]

def train(name):
    for lrG in lrGs:
        for bs in bss:
            os.system('python my_train.py --model_name %s --ep %d --lrG %f --bs %d' % (name, ep, lrG, bs))


model_name = 'AutoEncoderGen'
train(model_name)

model_name = 'AutoEncoderGen_no_dropout'
train(model_name)
