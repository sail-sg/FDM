# Fast Diffusion Model

This is an official PyTorch implementation of Fast Diffusion Model.

**Abstract**: *Despite their success in real data synthesis, diffusion models (DMs) often suffer from slow and costly training and sampling issues, limiting their broader applications. To mitigate this, we propose a Fast Diffusion Model (FDM) which improves the diffusion process of DMs through the lens of stochastic optimization to speed up both training and sampling. Specifically, we first find that the diffusion process of DMs accords with the stochastic optimization process of stochastic gradient descent (SGD) on a stochastic time-variant problem. Note that momentum SGD uses both the current gradient and an extra momentum, achieving more stable and faster convergence. We are inspired to introduce momentum into the diffusion process to accelerate both training and sampling. However, this comes with the challenge of deriving the noise perturbation kernel from the momentum-based diffusion process. To this end, we frame the momentum-based process as a Damped Oscillation system whose critically damped state---the kernel solution---avoids oscillation and thus has a faster convergence speed of the diffusion process. Empirical results show that our FDM can be applied to several popular DM frameworks, e.g. VP, VE, and EDM, and reduces their training cost by about $50\%$ with comparable image synthesis performance on CIFAR-10, FFHQ, and AFHQv2 datasets. Moreover, FDM  decreases  their sampling steps by about $3\times$ to achieve similar performance under the same deterministic samplers.*

## Requirements
All experiments were conducted using PyTorch 1.13.0, CUDA 11.7.1, and CuDNN 8.5.0. We strongly recommend to use the [provided Dockerfile](./Dockerfile) to build an image to reproduce our experiments.

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
