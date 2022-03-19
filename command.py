import os

ep = 50
lrGs = [0.0001]
bss = [40]
model_name = ['pic2pic', 'AutoEncoderGen', 'GAN', 'AutoEncoderGen_no_dropout']

data_path = 'fontdata'
script_path = 'my_train.py'

for n in model_name:
    for lrG in lrGs:
        for bs in bss:
            os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path "%s\"'
                      % (script_path, n, ep, lrG, bs, data_path))
