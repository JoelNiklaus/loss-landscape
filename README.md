# Visualizing the Loss Landscape of Neural Nets

This repository is a fork of the [original repository](https://github.com/tomgoldstein/loss-landscape) by the authors of the paper
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

We add simple and easy to use installation and running instructions.

An [interactive 3D visualizer](http://www.telesens.co/loss-landscape-viz/viewer.html) for loss surfaces has been provided by [telesens](http://www.telesens.co/2019/01/16/neural-network-loss-visualization/).

Given a network architecture and its pre-trained parameters, this tool calculates and visualizes the loss surface along random direction(s) near the optimal parameters.
The calculation can be done in parallel with multiple GPUs per node, and multiple nodes.
The random direction(s) and loss surface values are stored in HDF5 (`.h5`) files after they are produced.

## Setup

### Installation

Tested on Ubuntu 16.04.6 LTS with Conda 4.8.3.

#### Option 1
Run ``conda env create python=3.8 -f env.yml``

(created with ``conda env export -f env.yml --no-builds``)

#### Option 2
Run ``conda create python=3.8 --name loss_landscape --file env_explicit.txt``

(created with ``conda list --explicit > env_explicit.txt``)


#### Troubleshooting
If none of the above options work: 
Try to install the packages manually. 
The most important packages are listed in the section Environment.

#### Environment
- [pytorch](https://pytorch.org/)
- [openmpi](https://www.open-mpi.org/)
- [mpi4py](https://mpi4py.scipy.org/docs/usrman/install.html)
- [numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html)  
- [h5py](http://docs.h5py.org/en/stable/build.html#install)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [scipy](https://www.scipy.org/install.html)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)

## What exactly do I need to do to make it work?

1. If you have a new dataset: add a new folder ``datasets/{your_dataset_name}``.
2. Add you data to ``datasets/{your_dataset_name}/data``.
3. Add the model definitions to a file in ``datasets/{your_dataset_name}/models``.
4. Add your trained network to a file in ``datasets/{your_dataset_name}/trained_nets/{your_model_with_hyper_parameters}``.
5. Add a file ``data_loader.py`` in ``datasets/{your_dataset_name}`` and implement the method ``get_data_loaders()``. You can find documentation in [data_loader.py](datasets/cifar10/data_loader.py).
6. Add a file ``model_loader.py`` in ``datasets/{your_dataset_name}`` and implement the method ``load()``. Also add to the file a dictionary called ``models`` containing a mapping between the name of your model and the model function. You can find documentation in [model_loader.py](datasets/cifar10/model_loader.py).


### Examples for running it
#### Locally without GPU

Implicit (short version):
```shell script
python plot_surface.py --name test_plot --model resnet56 --dataset cifar10 --x=-1:1:51 --y=-1:1:51 --plot \
--model_file datasets/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7
```

Explicit (long version):
```shell script
python plot_surface.py --name test_plot --model resnet56 --dataset cifar10 --x=-1:1:51 --y=-1:1:51 --plot \
--model_file datasets/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn
```

#### On a server with 4 GPUs and 16 CPUs 
Implicit (short version):
```shell script
nohup python plot_surface.py --name test_plot --model init_baseline_vgglike --dataset cinic10 --x=-1:1:51 --y=-1:1:51 --plot \
--model_file datasets/cinic10/trained_nets/init_baseline_vgglike_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1_ngpu=4/model_10.t7 \
--cuda --ngpu 4 --threads 8 --batch_size 8192 > nohup.out &
```

Explicit (long version):
```shell script
nohup python plot_surface.py --name test_plot --model init_baseline_vgglike --dataset cinic10 --x=-1:1:51 --y=-1:1:51 --plot \
--model_file datasets/cinic10/trained_nets/init_baseline_vgglike_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1_ngpu=4/model_10.t7 \
--cuda --ngpu 4 --threads 8 --batch_size 8192 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn > nohup.out &
```

Please find the description of all the possible parameters in [plot_surface.py](loss_landscape/plot_surface.py).
More examples can be found in [plot_examples.sh](script/plot_examples.sh).

Make sure you do not use mpi when you run it on a single machine.

## Pretrained Models
The code accepts pre-trained PyTorch models for the CIFAR-10 and CINIC-10 datasets out of the box, but other datasets can also be added.
To load the pre-trained model correctly, the model file should contain `state_dict`, which is saved from the `state_dict()` method.
The default path for pre-trained networks is `cifar10/trained_nets`.
Some of the pre-trained models and plotted figures can be downloaded here:
- [VGG-9](https://drive.google.com/open?id=1jikD79HGbp6mN1qSGojsXOZEM5VAq3tH) (349 MB)
- [ResNet-56](https://drive.google.com/a/cs.umd.edu/file/d/12oxkvfaKcPyyHiOevVNTBzaQ1zAFlNPX/view?usp=sharing) (10 MB)
- [ResNet-56-noshort](https://drive.google.com/a/cs.umd.edu/file/d/1eUvYy3HaiCVHTzi3MHEZGgrGOPACLMkR/view?usp=sharing) (20 MB)
- [DenseNet-121](https://drive.google.com/a/cs.umd.edu/file/d/1oU0nDFv9CceYM4uW6RcOULYS-rnWxdVl/view?usp=sharing) (75 MB)

## Data preprocessing
The data pre-processing method used for visualization should be consistent with the one used for model training.
No data augmentation (random cropping or horizontal flipping) is used in calculating the loss values.


## Troubleshooting

libgfortran 4.0.0 does not seem to be compatible with linux. 
Make sure you don't update the dependencies to include this.

## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{visualloss,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```
