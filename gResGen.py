import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ep = 1001
lrGs = [0.001,0.002,0.005,0.01]
bss = [40, 8, 4]
lrD_rate_s = [1]
autogen_name = ['ResGen']
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

