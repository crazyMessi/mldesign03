import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ep = 1
lrGs = [0.0001,0.0002,0.005]
bss = [80]
lrD_rate_s = [1]
autogen_name = ['ResGen']
gan_name = ['GAN', 'pic2pic', 'ResGAN']
g_loss_func = ['L1', 'fixed_L1']
dropout = [0, 1]
channels = [1]

data_path = 'fontdata'
script_path = 'my_train.py'

for n in gan_name:
    for lrG in lrGs:
        for bs in bss:
            for d in dropout:
                for c in channels:
                    for lo in g_loss_func:
                        os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                  '--g_loss_func %s --dropout %d --channels %d'
                                  % (script_path, n, ep, lrG, bs, data_path, lo, d, c))

