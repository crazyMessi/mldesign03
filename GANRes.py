data_path = 'fontdata'
script_path = 'my_train.py'
import os


# 不变参数
gan_name = ['ResGAN22']
eps = [204]
lrGs = [0.001]
weight_pic = 50
dropout = 0
g_loss_func = ['fixed_L1']                              
discriminator = ['pixel']
n_block = [22]
dp_epoch = [10]
dg_rate = [2]
# 变参
lrG_d = [90,30]
bss = [8,4]
lrDs = [0.000005,0.000005]

for ep in eps:
    for n in gan_name:
        for lrG in lrGs:
            for lrD in lrDs:            
                for bs in bss:
                    for lo in g_loss_func:
                        for dis in discriminator:
                            for dgr in dg_rate:
                                for dpe in dp_epoch:
                                    for lgd in lrG_d:
                                        for nb in n_block:
                                            name = '%s_%d'%(n,lgd)
                                            com = 'python \"%s\" --model_name %s --ep %d --lrG %f --lrD %f --bs %d --data_path \"%s\" --g_loss_func %s --discriminator %s --dg_rate %d --dp_epoch %d --weight_pic %d --dropout %d --channels 1 --lrG_d %d --n_block %d' % (script_path, name, ep, lrG, lrD, bs, data_path, lo, dis, dgr, dpe, weight_pic, dropout, lgd,nb)                         
                                            os.system(com)
                                            
