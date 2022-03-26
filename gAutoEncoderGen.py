import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ep = 200 #1表示g
lrGs = [0.0001]
bss = [8]
lrD_rate_s = [1]
autogen_name = ['AutoEncoderGen_origin']
# gan_name = ['GAN', 'pic2pic', 'ResGAN']
g_loss_func = ['fixed_L1']
dropout = [0]
channels = [1]


data_path = 'fontdata'
script_path = 'my_train.py'

for n in autogen_name:
    for lrG in lrGs:
        for bs in bss:
            for d in dropout:
                for c in channels:
                    for lo in g_loss_func:
                        name = '%s'%(n)
                        os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                  '--g_loss_func %s --dropout %d --channels %d'
                                  % (script_path, name, ep, lrG, bs, data_path, lo, d, c))

