# Fast Diffusion Model

This is an official PyTorch implementation of Fast Diffusion Model.
## Results
Image synthesis performance (FID) under different million training images (Mimg) is as follows.
|  Dataset | Duration<br>(Mimg) |  EDM | EDM-FDM |  VP  | VP-FDM |   VE  | VE-FDM |
|:--------:|:---------------:|:----:|:-------:|:----:|:------:|:-----:|:------:|
| CIFAR10 |        50       | 5.76 |   2.17  | 2.74 |  2.74  | 49.47 |  10.01 |
| CIFAR10 |       100       | 1.99 |   1.93  | 2.24 |  2.24  |  4.05 |  3.26  |
| CIFAR10 |       150       | 1.92 |   1.83  | 2.19 |  2.13  |  3.27 |  3.00  |
| CIFAR10 |       200       | 1.88 |   1.79  | 2.15 |  2.08  |  3.09 |  2.85  |
|   FFHQ   |        50       | 3.21 |   3.27  | 3.07 |  12.49 | 96.49 |  93.72 |
|   FFHQ   |       100       | 2.87 |   2.69  | 2.83 |  2.80  | 94.14 |  88.42 |
|   FFHQ   |       150       | 2.69 |   2.63  | 2.73 |  2.53  | 79.20 |  4.73  |
|   FFHQ   |       200       | 2.65 |   2.59  | 2.69 |  2.43  | 38.97 |  3.04  |
|  AFHQv2  |        50       | 2.62 |   2.73  | 3.46 |  25.70 | 57.93 |  54.41 |
|  AFHQv2  |       100       | 2.57 |   2.05  | 2.81 |  2.65  | 57.87 |  52.45 |
|  AFHQv2  |       150       | 2.44 |   1.96  | 2.72 |  2.47  | 57.69 |  50.53 |
|  AFHQv2  |       200       | 2.37 |   1.93  | 2.61 |  2.39  | 57.48 |  47.30 |

Image synthesis performance (FID) under different inference cost on AFHQv2 with EDM sampler.
| NFE |  EDM | EDM-FDM |  VP  | VP-FDM |   VE  | VE-FDM |
|:---:|:----:|:-------:|:----:|:------:|:-----:|:------:|
|  25 | 2.78 |   2.32  | 2.88 |  2.59  | 61.04 |  48.29 |
|  49 | 2.39 |   1.93  | 2.64 |  2.41  | 57.59 |  47.49 |
|  79 | 2.37 |   1.93  | 2.61 |  2.39  | 57.48 |  47.30 |

Image synthesis performance (FID) under different inference cost on AFHQv2 with DPM-Solver++.
| NFE |  EDM | EDM-FDM |  VP  | VP-FDM |   VE  | VE-FDM |
|:---:|:----:|:-------:|:----:|:------:|:-----:|:------:|
|  25 | 2.60 |   2.09  | 2.99 |  2.64  | 59.26 |  49.51 |
|  49 | 2.42 |   1.98  | 2.79 |  2.45  | 59.16 |  48.68 |
|  79 | 2.39 |   1.95  | 2.78 |  2.42  | 58.91 |  48.66 |

## Preparing datasets
**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

**FFHQ:** Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
    --dest=datasets/ffhq-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz
```

**AFHQv2:** Download the updated [Animal Faces-HQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) (`afhq-v2-dataset`) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/afhqv2 \
    --dest=datasets/afhqv2-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/afhqv2-64x64.zip --dest=fid-refs/afhqv2-64x64.npz
```
## Training from scratch
Train FDM for class-conditional CIFAR-10 using 8 GPUs:
```.bash
# EDM-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp \
    --precond=fdm_edm --warmup_ite=200 

# VP-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --cres=1,2,2,2 \
    --precond=fdm_vp --warmup_ite=400

# VE-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ncsnpp --cres=1,2,2,2 \
    --precond=fdm_ve --warmup_ite=400 
```

Train FDM for unconditional FFHQ using 8 GPUs:
```.bash
# EDM-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output
    --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp \
    --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15 \
    --precond=fdm_edm --warmup_ite=800 --fdm_multipler=1

# VP-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp \
    --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15 \
    --precond=fdm_vp --warmup_ite=400 --fdm_multipler=1

VE-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/ffhq-64x64.zip --cond=0 --arch=ncsnpp \
    --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15 \
    --precond=fdm_ve --warmup_ite=400
```

Train FDM for unconditional AFHQv2 using 8 GPUs:
```.bash
# EDM-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output
    --data=datasets/afhqv2-64x64.zip --cond=0 --arch=ddpmpp \
    --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15 \
    --precond=fdm_edm --warmup_ite=400

# VP-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/afhqv2-64x64.zip --cond=0 --arch=ddpmpp \
    --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15 \
    --precond=fdm_vp --warmup_ite=400

VE-FDM
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
    --data=datasets/afhqv2-64x64.zip --cond=0 --arch=ncsnpp \
    --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15 \
    --precond=fdm_ve --warmup_ite=400
```

### Calculating FID
To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `fid.py`, replace `$PATH_TO_CHECKPOINT` with the path to the checkpoint:

```.bash
# Generate 50000 images 
torchrun --standalone --nproc_per_node=8 generate.py --outdir=fid \
    --seeds=0-49999 --subdirs --network=$PATH_TO_CHECKPOINT
# Calculate FID
torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid \
    --ref=fid-refs/cifar10-32x32.npz
```

Note that the generated images should be evaluated against the same reference dataset that the model was originally trained on. Please ensure to replace the `--ref` option with the correct one (*e.g.*, `fid-refs/ffhq-64x64.npz` or `fid-refs/afhqv2-64x64.npz`) to obtain the right FID score.