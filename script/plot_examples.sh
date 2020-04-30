# resnet20 on diufpc (4 GPUS, 16 CPUS)
nohup python plot_surface.py --name test_plot --model resnet20 --x=-1:1:51 --y=-1:1:51 \
--cuda --ngpu 4 --threads 8 --batch_size 4096 \
--model_file datasets/cifar10/trained_nets/resnet20_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=10/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot > nohup.out &

# resnet56 sgd
python plot_surface.py --name test_plot --model resnet56 --x=-1:1:51 --y=-1:1:51 \
--model_file datasets/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

# resnet56 no skip connections
python plot_surface.py --name test_plot --model resnet56 --x=-1:1:51 --y=-1:1:51 \
--model_file datasets/cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

# densenet 121
python plot_surface.py --name test_plot --model densenet121 --x=-1:1:51 --y=-1:1:51 \
--model_file datasets/cifar10/trained_nets/densenet121_sgd_lr=0.1_bs=64_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

# densenet 121 without mpi on diufpc29 (4 GPUS, 16 CPUS)
nohup python plot_surface.py --name test_plot --model densenet121 --x=-1:1:51 --y=-1:1:51 \
--cuda --ngpu 4 --threads 8 --batch_size 4096 \
--model_file datasets/cifar10/trained_nets/densenet121_sgd_lr=0.1_bs=64_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot > nohup.out &

# init baseline vgg like locally without cuda
python plot_surface.py --name test_plot --model init_baseline_vgglike --dataset cinic10 --x=-1:1:51 --y=-1:1:51 \
--model_file datasets/cinic10/trained_nets/init_baseline_vgglike_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1_ngpu=4/model_10.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

# init baseline vgg like on diufpc (4 GPUS, 16 CPUS)
nohup python plot_surface.py --name test_plot --model init_baseline_vgglike --dataset cinic10 --x=-1:1:51 --y=-1:1:51 \
--cuda --ngpu 4 --threads 8 --batch_size 8192 \
--model_file cinic10/trained_nets/init_baseline_vgglike_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=1_ngpu=4/model_10.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot > nohup.out &
