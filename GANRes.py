import os

data_path = 'fontdata'
script_path = 'my_train.py'

# 不变参数
gan_name = ['ResGAN']
eps = [202]
lrGs = [0.001]
bss = [8]
lrDs = [0.00005]
weight_pic = 50
dropout = 0
g_loss_func = ['fixed_L1']                              
discriminator = ['p_resnet18']
# 变参
dg_rate = [1]
dp_epoch = [10]

for ep in eps:
    for n in gan_name:
        for lrG in lrGs:
            for lrD in lrDs:            
                for bs in bss:
                    for lo in g_loss_func:
                        for dis in discriminator:
                            for dgr in dg_rate:
                                for dpe in dp_epoch:
                                    name = '%s_%s'%(n,dis)
                                    com = 'python \"%s\" --model_name %s --ep %d --lrG %f --lrD %f --bs %d --data_path \"%s\" --g_loss_func %s --discriminator %s --dg_rate %d --dp_epoch %d --weight_pic %d --dropout %d --channels 3' % (script_path, name, ep, lrG, lrD, bs, data_path, lo, dis, dgr, dpe, weight_pic, dropout)                         
                                    os.system(com)
                                    
