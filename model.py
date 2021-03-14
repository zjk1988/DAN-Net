import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import time
import h5py
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch.utils.data as Data
from PIL import Image
import torch.utils.data 
import matplotlib.pyplot as plt 
import torch.distributed as dist
import argparse

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.im_net = IM_net()
        self.se_net = SE_net()
        self.recon = RECON
    
    def forward(self,sub,LI_proj,mp,pre_CT,metal_trace):

        sinosub = self.se_net(sub.float(),mp.float()).cuda(args.local_rank)
        sino = (sinosub+LI_proj).mul(metal_trace)+LI_proj.mul(1-metal_trace)
        sino2im = self.recon(torch.transpose(sino,2,3)).cuda(args.local_rank)
        
        inputs_im = torch.cat((sino2im,pre_CT),dim=1)
        im = self.im_net(inputs_im.float(),pre_CT.float())
        return sino,sino2im,im
