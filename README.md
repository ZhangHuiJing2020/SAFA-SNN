#SAFA-SNN

## This is the official code repository for the project: SAFA-SNN: Sparsity-Aware On-Device Few-Shot Class-Incremental Learning with Fast-Adaptive Structure of Spiking Neural Network (ICLR 2026) [[paper](https://openreview.net/pdf?id=9jcB40wjk3)]

## Overview

<p align="center">
  <img src="img/framework.png" width="700">
</p>

<p align="center">
  <em>The overall framework of our proposed method SAFA-SNN.</em>
</p>



## Environment Required

python=3.10.9    
torch=2.5.1    
tqdm=4.64.1  
numpy=1.23.5  
cuda=12.2  
torchvision=0.20.1  
spikingjelly=0.0.0.0.14

## Run
```
python main.py --config jsons/safa.json
```

## Contact
If you have any questions about our work or this repository, please contact us by email.
[huijingzhang@zju.edu.cn](mailto:huijingzhang@zju.edu.cn)

## Acknowledge

We would like to thank the developers of the following open-source projects for their invaluable contributions:

[CEC](https://github.com/icoz69/CEC-CVPR2021)