import torch
import torch.utils.data
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

class myDataset(Dataset):
    def __init__(self, file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,file_path7,file_path8):
        if not os.path.isdir(file_path1):
            raise ValueError("input file_path is not a dir")
        self.file_path1 = file_path1
        self.Xma = os.listdir(file_path1)
        self.Xma.sort()

        if not os.path.isdir(file_path2):
            raise ValueError("input file_path is not a dir")
        self.file_path2 = file_path2
        self.mp = os.listdir(file_path2)
        self.mp.sort()

        if not os.path.isdir(file_path3):
            raise ValueError("input file_path is not a dir")
        self.file_path3 = file_path3
        self.metal_trace = os.listdir(file_path3)
        self.metal_trace.sort()

        if not os.path.isdir(file_path4):
            raise ValueError("input file_path is not a dir")
        self.file_path4 = file_path4
        self.ma_proj = os.listdir(file_path4)
        self.ma_proj.sort()     

        if not os.path.isdir(file_path5):
            raise ValueError("input file_path is not a dir")
        self.file_path5 = file_path5
        self.LI_proj = os.listdir(file_path5)
        self.LI_proj.sort() 
        
        if not os.path.isdir(file_path6):
            raise ValueError("input file_path is not a dir")
        self.file_path6 = file_path6
        self.gt_CT = os.listdir(file_path6)
        self.gt_CT.sort()     

        if not os.path.isdir(file_path7):
            raise ValueError("input file_path is not a dir")
        self.file_path7 = file_path7
        self.gt_proj = os.listdir(file_path7)
        self.gt_proj.sort() 

        if not os.path.isdir(file_path8):
            raise ValueError("input file_path is not a dir")
        self.file_path8 = file_path8
        self.mask = os.listdir(file_path8)
        self.mask.sort() 
        
     
    def __getitem__(self, index):
        
        f = os.path.join(self.file_path1, self.Xma[index])
        data = loadmat(f)
        data = data['pre_CT']
        data = np.expand_dims(data, axis=0)
        ma_CT = data
        

        f = os.path.join(self.file_path2, self.mp[index])
        data = h5py.File(f,'r')
        data = data['mp'][:].T
        data = np.expand_dims(data, axis=0)
        mp = data
        
        f = os.path.join(self.file_path3, self.metal_trace[index])
        data = h5py.File(f,'r')
        data = data['metal_trace'][:].T
        data = np.expand_dims(data, axis=0)
        metal_trace = data
        
        
        f = os.path.join(self.file_path4, self.ma_proj[index])
        data = loadmat(f)
        data = data['pre_proj']
        data = np.expand_dims(data, axis=0)
        ma_proj = data
      
        f = os.path.join(self.file_path5, self.LI_proj[index])
        data = h5py.File(f,'r')
        data = data['LI_proj'][:].T
        data = np.expand_dims(data, axis=0)
        LI_proj = data
      
        f = os.path.join(self.file_path6, self.gt_CT[index])
        data = h5py.File(f,'r')
        data = data['gt_CT'][:].T
        data = np.expand_dims(data, axis=0)
        gt_CT = data
        
        f = os.path.join(self.file_path7, self.gt_proj[index])
        data = h5py.File(f,'r')
        data = data['gt_proj'][:].T
        data = np.expand_dims(data, axis=0)
        gt_proj = data
        
        f = os.path.join(self.file_path8, self.mask[index])
        data = h5py.File(f,'r')
        data = data['metal_im'][:].T
        data = np.expand_dims(data, axis=0)
        mask = data

        return ma_CT,mp,ma_proj,gt_CT,gt_proj,mask,metal_trace,LI_proj

    def __len__(self):
        return len(self.Xma)
    
train_dataset = myDataset(file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,file_path7,file_path8)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

# file_path1 = "pre_CT/"
# file_path4 = 'pre_proj/'
# file_path3 = 'metal_trace/'
# file_path2 = 'mp_train/'
# file_path5 = 'LI_proj/'    
# file_path6 = 'gt_CT/'
# file_path7 = 'gt_proj/'  
# file_path8 = 'mask_im/' 