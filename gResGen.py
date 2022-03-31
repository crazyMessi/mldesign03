import os
ep = 200
lrGs = [0.001]
bss = [8]
model_name = ['ResGen']
g_loss_func = ['fixed_L1']

data_path = 'fontdata'
script_path = 'my_train.py'

for n in model_name:
    for lrG in lrGs:
        for bs in bss:
            for lo in g_loss_func:
                    name = '%s_d%d'%(n)
                    os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                '--g_loss_func %s --n_downsampling %d'
                                % (script_path, name, ep, lrG, bs, data_path, lo))

