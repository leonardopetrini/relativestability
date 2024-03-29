## Reproducing figures of the paper *How deep convolutional neural networks lose spatial information with training*

Experiments are run using [`grid`](https://github.com/mariogeiger/grid/tree/master/grid).

The [`diffeo-sota`](https://github.com/leonardopetrini/diffeo-sota) repository implements the training of deep neural networks.

The [`relativestability`](https://github.com/leonardopetrini/relativestability) repository implements the diffeomorphisms (D_f), noise (G_f) and relative (R_f) stabilities computation. 
Experiments of Section 4 are run using [`scale_detection_1d`](https://github.com/UmbertoTomasini/scale_detection_1d). 

An PyTorch implementation of the [`scale-detection task` in 2D can be found here](https://github.com/leonardopetrini/diffeo-sota/blob/main/datasets/twopoints.py).

### Figures 1, 10
Networks trained on CIFAR10 at different added noise intensities:
- CNNs (`diffeo-sota` repo)
```
python -m grid /scratch/izar/lpetrini/results/corrupted_cifar --n 6 "
grun python main.py --batch_size 128 --epochs 250 --save_best_net 1 --diffeo 0 --random_crop 1 --hflip 1 --ptr 50000
 " --seed_init 0 1 2 3 --net:str 'VGG11' 'ResNet34' 'ResNet50' 'VGG11bn' 'VGG16bn' 'VGG19bn' 'AlexNet' 'LeNet' 'ResNet18' 'EfficientNetB0' --dataset:str 'cifar10' --gaussian_corruption_std 0. 1e-2 1e-1 1 10 30 1e2
```
- FCNs (`diffeo-sota` repo)
```
python -m grid /scratch/izar/lpetrini/results/corrupted_cifar --n 6 "
grun python main.py --batch_size 128 --epochs 250 --save_best_net 1 --diffeo 0 --random_crop 1 --hflip 1 --ptr 50000 --weight_decay 0
 " --seed_init 0 1 2 3 4 5 6 7 8 9 10 --net:str 'DenseNetL4' 'DenseNetL2' 'DenseNetL6' --optim:str 'adam' --scheduler:str 'none' --dataset:str 'cifar10' --gaussian_corruption_std 0. 1e-2 1e-1 1 10 30 1e2
```

Stabilities are then computed with (`relativestability` repo)
```
python -m grid /scratch/izar/lpetrini/results/corrupted_cifar_stab --n 6 "
grun python main.py --init 0 --init_samples 5 --dataset cifar10 --filename noise_corruption2 --by_layer 0 --P 500
" --net:str 'DenseNetL4' 'DenseNetL2' 'DenseNetL6' 'VGG11' 'ResNet34' 'ResNet50' 'VGG11bn' 'VGG16bn' 'VGG19bn' 'AlexNet' 'LeNet' 'ResNet18' 'EfficientNetB0' --shuffle_channels 0 --corrupt_test 1 --gaussian_corruption_std 0. 1e-2 1e-1 1 10 30 1e2
```

### Figures 3, 11
Uses the networks trained for Fig. 1, `by_layer = 1` computes stabilities layer by layer (`relativestability` repo):
```
python -m grid /scratch/izar/lpetrini/results/corrupted_cifar_stab --n 6 "
grun python main.py --init 0 --dataset cifar10 --by_layer 1 --P 500
" --net:str 'VGG11' 'VGG11bn' 'AlexNet' 'LeNet' --shuffle_channels 0 1 --corrupt_test 0 --gaussian_corruption_std 0.
```

### Figure 8
Training of deep networks on the scale-detection task, `diffeo-sota` repo:
```
python -m grid /home/lpetrini/results/scaledetection --n 6 "
grun python main.py --batch_size 128 --save_best_net 1 --diffeo 0 --random_crop 0 --hflip 0 --lr .05 --epochs 300 --norm L2
 " --seed_init 0 1 2 3 4 5 6 7 8 9 --dataset:str 'twopoints' --ptr 1024 2048 --xi 14 --bias 0 --gap 2 --loss:str 'hinge' --net:str 'VGG13bn' 'VGG11' 'ResNet34' 'ResNet50' 'VGG11bn' 'VGG16bn' 'VGG19bn' 'AlexNet' 'LeNet' 'ResNet18' 'EfficientNetB0'

```

### Figures 4, 12, 13, 14
(`relativestability` repo)

Uses the networks trained for Fig. 1 and 5. Projections of the network weights on the grid-Laplacian eigenvectors are computed using `laplacian_prog` from `laplacian`.

### Figure 9
Uses the networks trained for Fig. 5, `by_layer = 1` computes stabilities layer by layer (`relativestability` repo):
```
python -m grid /scratch/izar/lpetrini/results/scaledetection_stab --n 6 "
grun python main.py --init 0 --dataset twopoints --by_layer 1 --P 500
" --net:str 'VGG11' 'VGG11bn' 'VGG13bn' --shuffle_channels 0 1
```

### Figure 7
See `scale_detection_1d` repository for details on how to reproduce the relative experiments.
