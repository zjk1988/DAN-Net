import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import time
import h5py
from glob import glob
import numpy as np

class down(nn.Module):
    def __init__(self,in_c,o_c):
        super(down,self).__init__()
        self.d = nn.Sequential(
                nn.Conv2d(in_c,o_c,4,2,1),
                nn.BatchNorm2d(o_c),
                nn.LeakyReLU(0.2, True)
                )
    def forward(self,x):
        out = self.d(x)
        return out
class downfirst(nn.Module):
    def __init__(self,in_c,o_c):
        super(downfirst,self).__init__()
        self.d = nn.Sequential(
                nn.Conv2d(in_c,o_c,4,2,1),
                nn.BatchNorm2d(o_c),
                )
    def forward(self,x):
        out = self.d(x)
        return out
class up(nn.Module):
    def __init__(self,in_c,o_c,size):
        super(up,self).__init__()
        self.up = nn.Sequential(
                nn.Conv2d(in_c,o_c,3,1,1),
                nn.BatchNorm2d(o_c),
                nn.ReLU()
                )    
        self.size = size
    def forward(self,x):
        out = nn.functional.interpolate(x,size=self.size,mode='bilinear',align_corners =True)
        out = self.up(out)
        return out
class uplast(nn.Module):
    def __init__(self,in_c,o_c,size):
        super(uplast,self).__init__()
        self.up = nn.Sequential(
                nn.Conv2d(in_c,o_c,3,1,1),
                nn.Tanh()
                )    
        self.size = size
    def forward(self,x):
        out = nn.functional.interpolate(x,size=self.size,mode='bilinear',align_corners =True)
        out = self.up(out)
        return out

class SE_net(nn.Module):
    def __init__(self):
        super(SE_net,self).__init__()
        n1 = 64
        filters = [n1,n1*2,n1*4,n1*8,n1*8]
        self.pool1 = nn.AvgPool2d(2,2)
        self.pool2 = nn.AvgPool2d(2,2)
        self.pool3 = nn.AvgPool2d(2,2)
        self.pool4 = nn.AvgPool2d(2,2)
        self.pool5 = nn.AvgPool2d(2,2)
        
        self.d0 = downfirst(2,filters[0])
        self.d1 = down(filters[0]+1,filters[1])
        self.d2 = down(filters[1]+1,filters[2])
        self.d3 = down(filters[2]+1,filters[3])
        self.d4 = down(filters[3]+1,filters[4])
        
        self.up5 = up(filters[4]+1,filters[3],size=[22,22])
        self.up4 = up(filters[3]+filters[3]+1,filters[2],size=[45,45])
        self.up3 = up(filters[2]+filters[2]+1,filters[1],size=[91,90])
        self.up2 = up(filters[1]+filters[1]+1,filters[0],size=[183,180])
        self.up1 = uplast(filters[0]+filters[0]+1,1,size = [367,361])
        
    def forward(self,x,mask):
        e0 = torch.cat((x,mask),dim=1)
        e0 = self.d0(e0)
        mask2 = self.pool1(mask)
        e0 = torch.cat((e0,mask2),dim=1)
        e1 = self.d1(e0)
        mask3 = self.pool2(mask2)
        e1 = torch.cat((e1,mask3),dim=1)
        e2 = self.d2(e1)
        mask4 = self.pool3(mask3)
        e2 = torch.cat((e2,mask4),dim=1)
        e3 = self.d3(e2)
        mask5 = self.pool4(mask4)
        e3 = torch.cat((e3,mask5),dim=1)
        e4 = self.d4(e3)
        mask6 = self.pool5(mask5)
        e4 = torch.cat((e4,mask6),dim=1)
        
        d5 = self.up5(e4)
        d5 = torch.cat((d5,e3),dim=1)
        d4 = self.up4(d5)
        d4 = torch.cat((d4,e2),dim=1)
        d3 = self.up3(e4)
        d3 = torch.cat((d3,e1),dim=1)
        d2 = self.up2(d3)
        d2 = torch.cat((d2,e0),dim=1)
        d1 = self.up1(d2)
        
        return d1