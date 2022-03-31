import os
data_path = 'fontdata'
script_path = 'my_train.py'

ep = 202
lrGs = [0.0001,0.0002,0.00005]
bss = [8]
lrDs = [0.0001,0.00005]
#autogen_name = ['AutoEncoderGen', 'UNetGen', 'ResGen']
gan_name = ['GAN']
g_loss_func = ['fixed_L1']
dropout = [0, 1]
channels = [1]
wight_pic = [10, 50]


for n in gan_name:
    for lrG in lrGs:
        for lrD in lrDs:
            for bs in bss:
                for d in dropout:
                    for c in channels:
                        for lo in g_loss_func:
                            for lamb in wight_pic:
                                name = '%s_%d_%d'%(n,d,lamb)
                                os.system('python \"%s\" --model_name %s --ep %d --lrG %f  --lrD %f --bs %d --data_path \"%s\" '
                                        '--g_loss_func %s --dropout %d --channels %d --weight_pic %d'
                                        % (script_path, name, ep, lrG, lrD, bs, data_path, lo, d, c, lamb))

