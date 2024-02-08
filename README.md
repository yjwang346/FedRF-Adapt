# FedRF-Adapt: Robust and Communication-Efficient Federated Domain Adaptation via Random Features
This repository contains code to reproduces the numerical results in the paper "FedRF-Adapt: Robust and Communication-Efficient Federated Domain Adaptation via Random Features"

## About the code
- The repository `/FedRF-Adapt` contains
    - v1_digit_five_multi_source_train.py: FedRF-Adapt experiments on digit-five dataset; 
    - v1_office_multi_source_train.py: FedRF-Adapt experiments on office dataset; 
    - v1_visda_separate_source_target_train.py: FedRF-Adapt experiments on visda dataset; 
    - v1_FedRF_Adapt_model.py: FedRF-Adapt with different MMD alignment strategies based on different features; 
    - linear_classifier.py: classifier module of the model; 
    - TorchRandomF.py: random feature-based MMD (Maximum Mean Discrepancy) algorithm based on torch.

## Dependencies
- python=3.8
- pytorch=2.0.1, pytorch-cuda=11.7
- torchvision=0.15.2

## Contact Information
For questions, feel free to reach out to authors: 
- Yuanjei Wang: [m202172546@hust.edu.cn](mailto:m202172546@hust.edu.cn)
- Zhanbo Feng: [zhanbofeng@sjtu.edu.cn](mailto:zhanbofeng@sjtu.edu.cn)
- Zhenyu Liao: [zhenyu_liao@hust.edu.cn](mailto:zhenyu_liao@hust.edu.cn)
