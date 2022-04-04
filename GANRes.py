import os
data_path = 'fontdata'
script_path = 'my_train.py'

eps = [202, 204, 206]
lrGs = [0.002]
bss = [8]
lrDs = [0.0002] 
gan_name = ['ResGAN']
g_loss_func = ['fixed_L1']                              
discriminator = ['pixel']
dg_rate = [1]
dp_epoch = [0]

for ep in eps:
    for n in gan_name:
        for lrG in lrGs:
            for lrD in lrDs:            
                for bs in bss:
                    for lo in g_loss_func:
                        for dis in discriminator:
                            for dgr in dg_rate:
                                for dpe in dp_epoch:
                                    name = '%s'%(n)
                                    com = 'python \"%s\" --model_name %s --ep %d --lrG %f --lrD %f --bs %d --data_path \"%s\" --g_loss_func %s --discriminator %s --dg_rate %d --dp_epoch %d --weight_pic 50 --dropout 0' % (script_path, name, ep, lrG, lrD, bs, data_path, lo, dis, dgr, dpe)                         
                                    os.system(com)
                                    
