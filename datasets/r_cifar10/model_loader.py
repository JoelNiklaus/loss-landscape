import os

import torch
import datasets.r_cifar10.models.BabyDenseNet as BabyDenseNet


import sys
sys.path.insert(0, '~/gale/')


# map between model name and function
models = {
    'fatbabydensenet21'          : BabyDenseNet.fatbabydensenet21,
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name](num_classes=10)
    if data_parallel: # the model is saved in data parallel mode
        net = torch.nn.DataParallel(net)

    if model_file:
        try:
            assert os.path.exists(model_file), model_file + " does not exist."
        except:
            import IPython; IPython.embed();
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
