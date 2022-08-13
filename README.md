# EAdderSR: Enhanced AdderSR for Single Image Super Resolution
Jie Song, Huawei Yi, Wenqian Xu, Xiaohui Li, Bo Li, Yuanyuan Liu
> Reference Toolbox: Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR. https://github.com/xinntao/BasicSR, 2020.


1. Put trainset and testset into /datasets.

2. Put vgg19.pth into basicsr\models\archs\vgg19 checkpoint.

3. Train (MN):

    ```
    python basicsr/train.py -opt options\train\SRResNet_SRGAN_EAdderSR\train_MSRResNet_EAdderSR_x4.yml 
   
    ``` 
4. Train (MN + CGKD):

    ```
    python basicsr/train_CGKD.py -opt options\train\SRResNet_SRGAN_EAdderSR\train_MSRResNet_EAdderSR_x4.yml 
   
    ```
