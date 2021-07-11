# DAN-Net
------
paper:DAN-Net: [Dual-Domain Adaptive-Scaling Non-local Network for CT Metal Artifact Reduction](https://arxiv.org/abs/2102.08003)

by Tao Wang [scuer_wt@scu.stu.edu.cn](scuer_wt@scu.stu.edu.cn).
This repository implements DAN-Net by Pytorch for metal artifacts reduction (MAR) in CT images. We re-implement the reconstruction layer according to the formulas in DuDoNet and we used Numba in Python to achieve parallel computing.

This work was accepted by MICCAI2021, and the extended version was accepted by Physics in Medicine & Biology (PMB).

Prerequisites
-------------

This repository needs the following system settings:

 - Python 3.6 
 - Pytorch 1.6.0
 - CUDA 10.1
 - Matlab R2017b
 

