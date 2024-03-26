## Efficient Statistical Sampling Adaptation for Exemplar-Free Class Incremental Learning
This is the implementation of the paper "Efficient Statistical Sampling Adaptation for Exemplar-Free Class Incremental Learning" (T-CSVT).

### Requirements
- Python 3.8
- PyTorch 1.8.1
- cuda 11.2

### Datasets
Download following datasets:
> #### CIFAR-100
> #### TinyImageNet
> #### ImageNet-Subset
Locate the above three datasets under ./dataset directory.

### Parameter settings
> #### The implementation details are in the body of the paper.

### Training
> #### 1. Download pretrained models to ./pre folder.
> #### 2. Training
> ```bash
> sh train_main.sh 
> ```

## Requirements
> We thank the following repos providing helpful components/functions in our work.
- [SSRE](https://github.com/zhukaii/SSRE)
