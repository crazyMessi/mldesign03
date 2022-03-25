import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ep = 1001
lrGs = [0.0025,0.002,0.0015]
bss = [80, 40, 20, 8]
lrD_rate_s = [1]
autogen_name = ['UNetGen']
# gan_name = ['GAN', 'pic2pic', 'ResGAN']
g_loss_func = ['fixed_L1','L1']
dropout = [0, 1]
channels = [1]


data_path = 'fontdata'
script_path = 'my_train.py'

for n in autogen_name:
    for lrG in lrGs:
        for bs in bss:
            for d in dropout:
                for c in channels:
                    for lo in g_loss_func:
                        name = '%s_%d_%d_%s'%(n,d,c,lo)
                        os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                  '--g_loss_func %s --dropout %d --channels %d'
                                  % (script_path, name, ep, lrG, bs, data_path, lo, d, c))

