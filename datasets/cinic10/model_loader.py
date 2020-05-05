import os

import torch
import datasets.cinic10.models.baselines as baselines

# map between model name and function
models = {
    'spectral_baseline'          : baselines.SpectralBaselineDef,
    'init_baseline'              : baselines.InitBaselineDef,
    'init_baseline_vgglike'      : baselines.InitBaselineVGGLikeDef,
    'lda_simple'                 : baselines.LDA_SimpleDef,
}


def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel:  # the model is saved in data parallel mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
