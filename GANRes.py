import os

from torch import dropout
data_path = 'fontdata'
script_path = 'my_train.py'

ep = 202
lrGs = [0.001]
bss = [8]
lrDs = [0.00005]

discriminator = ['patch']
gan_name = ['ResGAN']

g_loss_func = ['fixed_L1']
lrDs = [0.005]                               
discriminator = ['pixel']

for n in gan_name:
    for lrG in lrGs:
        for lrD in lrDs:            
            for bs in bss:
                for lo in g_loss_func:
                    for dis in discriminator:
                        for drp in dropout:
                            name = '%s_%s_drp_%d'%(n,dis,drp)
                            os.system('python \"%s\" --model_name %s --ep %d --lrG %f --lrD %f --bs %d --data_path \"%s\" '
                                    '--g_loss_func %s --discriminator %s --dropout %d' 
                                    % (script_path, name, ep, lrG, lrD, bs, data_path, lo, dis,drp))
                                
