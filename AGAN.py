import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ep = 500
lrGs = [0.001,0.002,0.005]
bss = [40, 20, 10, 8]
lrD_rate_s = [1]
#autogen_name = ['AutoEncoderGen', 'UNetGen', 'ResGen']
gan_name = ['GAN']
g_loss_func = ['fixed_L1']
dropout = [0, 1]
channels = [1]
wight_pic = [10, 50]


data_path = 'fontdata'
script_path = 'my_train.py'

for n in gan_name:
    for lrG in lrGs:
        for bs in bss:
            for d in dropout:
                for c in channels:
                    for lo in g_loss_func:
                        for lamb in wight_pic:
                            name = '%s_%d_%d'%(n,d,lamb)
                            os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                    '--g_loss_func %s --dropout %d --channels %d --weight_pic %d'
                                    % (script_path, name, ep, lrG, bs, data_path, lo, d, c, lamb))

