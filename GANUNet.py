import os
data_path = 'fontdata'
script_path = 'my_train.py'

# 不变参数
gan_name = ['pic2pic']
lrGs = [0.002]
bss = [4]
weight_pic = 20
g_loss_func = ['fixed_L1']                              
discriminator = ['pixel']
dg_rate = [2]
dp_epoch = [0]
dropout = 1
lrDs = [0.00005]

# 变参
eps = [202,204]

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
                                    com = 'python \"%s\" --model_name %s --ep %d --lrG %f --lrD %f --bs %d --data_path \"%s\" --g_loss_func %s --discriminator %s --dg_rate %d --dp_epoch %d --weight_pic %d --dropout %d' % (script_path, name, ep, lrG, lrD, bs, data_path, lo, dis, dgr, dpe, weight_pic, dropout)                         
                                    os.system(com) 
                                
