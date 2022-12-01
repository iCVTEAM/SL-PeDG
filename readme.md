This repository contains PyTorch codes for the ACM MM 2022 paper "[**Revisiting Stochastic Learning for Generalizable Person Re-identification**](https://dl.acm.org/doi/abs/10.1145/3503161.3547812)"

## Set up with Conda
```
conda env create -f reid.yml
conda activate reid
pip install -r reid.txt
```

## Training and Evaluating
### Dataset
Train datasets and evaluation datasets should be download in the directory *datasets*. The download way is shown in the [MetaBIN](https://github.com/bismex/MetaBIN).

### Train or evaluate
Download [pretrained model](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth) into the directory *pretrain*. 

Download [trained models](https://drive.google.com/drive/folders/1OK-h8dOd7SkXv6klh5VrW6PaIBCeKNvE?usp=share_link) into the directory *model_weights*. 

```
# Train or evaluate in the C2-C3-D-M-CS setting
sh train1.sh
sh eval1.sh

# Train or evaluate in the C3-D-M-MT setting
sh train2.sh
sh eval2.sh
```

## Others
The implement of proposed stochastic sampler in *./fastreid/data/samplers/triplet_sampler.py/DomainSplitBalancedSampler*

The implement of proposed gradient dropout in *./fastreid/engine/hooks.py/DropoutSGDHook* and *./fastreid/solver/optim/dropout_sgd.py/DropoutSGD*

## Citation
```
@InProceedings{Zhao_2022_ACMMM,
    author    = {Zhao, Jiajian and Zhao, Yifan and Chen, Xiaowu and Li, Jia},
    title     = {Revisiting Stochastic Learning for Generalizable Person Re-identification},
    booktitle = {Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)},
    month     = {October},
    year      = {2022},
    pages     = {1758-1768}
}
```

## Acknowledgment
This repository is based on the implementation of [fast-reid](https://github.com/JDAI-CV/fast-reid).
