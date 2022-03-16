import os

ep = 1
lrGs = [0.001, 0.0001, 0.00001]
bss = [8, 16, 32]

for lrG in lrGs:
    for bs in bss:
        os.system('python my_train.py --model_name AutoEncoderGen --ep %d --lrG %f --bs %d' % (ep, lrG, bs))