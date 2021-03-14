import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import h5py
import numpy as np

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)





import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
class conv_block1(nn.Module):
    def __init__(self,in_c,o_c):
        super(conv_block1,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.ReLU(),
                nn.Conv2d(o_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.ReLU()
                )
    def forward(self,x):
        out = self.conv(x)
        return out

class conv_block(nn.Module):
    def __init__(self,in_c,o_c):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=4,stride=2, padding=3, dilation=2),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.LeakyReLU(0.2, True),
                # nn.Conv2d(o_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                # nn.BatchNorm2d(o_c),
                # nn.LeakyReLU(0.2, True),

                )
    def forward(self,x):
        out = self.conv(x)
        return out
class up_conv(nn.Module):
    def __init__(self,in_c,o_c,size):
        super(up_conv,self).__init__()
        self.size = size
        self.up = nn.Sequential(
                # nn.Upsample(size=size,mode='bilinear',align_corners =True),
                nn.Conv2d(in_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.LeakyReLU(0.2, True),
                )
    def forward(self,x):
        o = F.interpolate(x,size=self.size,mode='bilinear',align_corners =True)
        out = self.up(o)
        return out
class IM_net(nn.Module):
    def __init__(self,in_c=2,o_c=1):
        super(IM_net,self).__init__() 
        n1 = 32
        filters = [n1,n1*2,n1*4,n1*8,n1*16]    
      
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_c,filters[0],kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(filters[0]),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(filters[0],filters[0],kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(filters[0]),
                nn.LeakyReLU(0.2, True),
                )
        self.conv2 = conv_block(filters[0],filters[1])
        self.conv3 = conv_block(filters[1],filters[2])
        self.conv4 = conv_block(filters[2],filters[3])
        self.conv5 = conv_block(filters[3],filters[4])
      
        self.Up5 = up_conv(filters[4],filters[3],size=[32,32])
        self.up_conv5 = conv_block1(filters[4],filters[3])
      
        self.Up4 = up_conv(filters[3],filters[2],size=[64,64])
        self.up_conv4 = conv_block1(filters[3],filters[2])

        self.Up3 = up_conv(filters[2],filters[1],size=[128,128])
        self.up_conv3 = conv_block1(filters[2],filters[1])
      
        self.Up2 = up_conv(filters[1],filters[0],size=[256,256])
        self.up_conv2 = conv_block1(filters[1],filters[0])
      
        self.conv_f5 = nn.Conv2d(filters[4],o_c,kernel_size=1,stride=1,padding=0)
        self.conv_f4 = nn.Conv2d(filters[3],o_c,kernel_size=1,stride=1,padding=0)
        self.conv_f3 = nn.Conv2d(filters[2],o_c,kernel_size=1,stride=1,padding=0)
        self.conv_f2 = nn.Conv2d(filters[1],o_c,kernel_size=1,stride=1,padding=0)

        self.conv_f1 = nn.Conv2d(filters[0],o_c,kernel_size=1,stride=1,padding=0)
      
        self.non2 = NONLocalBlock2D(filters[1])
        self.non3 = NONLocalBlock2D(filters[2])

    def forward(self,x,batch_has_m1):
        e1 = self.conv1(x)
#        e1 = self.non1(e1)
        e2 = self.conv2(e1)
        e2 = self.non2(e2)
        
        e3 = self.conv3(e2)
        e3 = self.non3(e3)
        
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4,d5),dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2,d3),dim=1)
        d3 = self.up_conv3(d3)
     
        d2 = self.Up2(d3)
        d2 = torch.cat((e1,d2),dim=1)
        d2 = self.up_conv2(d2)
  
        out1 = self.conv_f1(d2)
        out = out1 + batch_has_m1

        return out