"""
Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
"""
import logging
# from models.registry import Model

import torch
import torch.nn as nn
import torch.nn.functional as F


class BabyDenseNet(nn.Module):
    r"""
    Densenet model class, based on "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    It is better suited for smaller images as the expected input size is 3x32x32 (CxHxW).

    Attributes
    ----------
    ablate : bool
        If true, return output of model before applying final classification layer.
    conv1 : torch.nn.Sequential
        First block of conv,BN and ReLu that brings input from 32x32 to 28x28.
    denseblock2 : torch.nn.Sequential
    denseblock3 : torch.nn.Sequential
    denseblock4 : torch.nn.Sequential
        groups of "layers" for the network. Each of them is composed of multiple
        sub-blocks depending on the type of DenseNet.
    bn5: torcch.nn.nBatchNorm2d
        BatchNorm layer
    classifier : torch.nn.Linear
        Final classification layer that takes features as input and produces
        classification values.
    total_parameters : int
        Number of parameters in the network.
    """
    expected_input_size = (32, 32)

    def __init__(self, num_classes, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 ablate=False, **kwargs):

        super(BabyDenseNet, self).__init__()

        self.ablate = ablate
        self.num_features = num_init_features  # Attention: this gets updated after each denseblock creation!

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.num_features, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(inplace=True),
        )


        self.denseblock2 = self._make_block(block_config[1], self.num_features, bn_size, growth_rate, drop_rate)
        self.denseblock3 = self._make_block(block_config[2], self.num_features, bn_size, growth_rate, drop_rate)
        self.denseblock4 = self._make_block(block_config[3], self.num_features, bn_size, growth_rate, drop_rate,
                                            add_trans=False)

        # Final batch norm
        self.bn5 = nn.BatchNorm2d(self.num_features)

        # Linear layer
        if not self.ablate:
            self.classifier = nn.Linear(self.num_features, num_classes)

        # Compute and log the number of parameters of the network
        self.total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f'Total number of parameters is {self.total_parameters / 10 ** 6}M')

    def forward(self, x):
        features = self.conv1(x)
        features = self.denseblock2(features)
        features = self.denseblock3(features)
        features = self.denseblock4(features)
        features = self.bn5(features)

        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        if self.ablate:
            return out
        else:
            out = self.classifier(out)
            return out

    def _make_block(self, num_layers, num_features, bn_size, growth_rate, drop_rate, add_trans=True):
        block = _DenseBlock(num_layers=num_layers, num_input_features=self.num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_features = self.num_features + num_layers * growth_rate
        if add_trans:
            trans = _Transition(num_input_features=self.num_features, num_output_features=self.num_features // 2)
            block.add_module('transition', trans)
            self.num_features = self.num_features // 2
        return block

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# @Model
def babydensenet11(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=64, growth_rate=32, block_config=(1, 1, 1, 1),
                         **kwargs)
    return model
babydensenet11.expected_input_size = BabyDenseNet.expected_input_size

# @Model
def fatbabydensenet11(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=256, growth_rate=320, block_config=(1, 1, 1, 1),
                         **kwargs)
    return model
fatbabydensenet11.expected_input_size = BabyDenseNet.expected_input_size


# @Model
def babydensenet21(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=64, growth_rate=32, block_config=(1, 2, 4, 2),
                         **kwargs)
    return model
babydensenet21.expected_input_size = BabyDenseNet.expected_input_size

# @Model
def fatbabydensenet21(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=256, growth_rate=176, block_config=(1, 2, 4, 2),
                         **kwargs)
    return model
fatbabydensenet21.expected_input_size = BabyDenseNet.expected_input_size


# @Model
def babydensenet115(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=64, growth_rate=32, block_config=(0, 16, 24, 16),
                         **kwargs)
    return model
babydensenet115.expected_input_size = BabyDenseNet.expected_input_size


# @Model
def babydensenet155(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=96, growth_rate=48, block_config=(0, 16, 36, 24),
                         **kwargs)
    return model
babydensenet155.expected_input_size = BabyDenseNet.expected_input_size


# @Model
def babydensenet163(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=64, growth_rate=32, block_config=(0, 16, 32, 32),
                         **kwargs)
    return model
babydensenet163.expected_input_size = BabyDenseNet.expected_input_size


# @Model
def babydensenet195(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BabyDenseNet(num_init_features=64, growth_rate=32, block_config=(0, 16, 48, 32),
                         **kwargs)
    return model
babydensenet195.expected_input_size = BabyDenseNet.expected_input_size
