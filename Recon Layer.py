import numpy as np
import torch
import math
from scipy.io import loadmat,savemat
from scipy import interpolate
import time
from numba import jit,vectorize
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#######################################
##forward

from numba import cuda,float32

@cuda.jit
def func(alpha,proj_ifft,rec):            
    x,y = cuda.grid(2)
    for i in range(361):
        theta = alpha[i]
        pos = (x-128)*math.cos(theta)+(y-128)*math.sin(theta)+183
        temp1 = proj_ifft[i,int(math.floor(pos))]
        temp2 = proj_ifft[i,int(math.ceil(pos))]
        temp = (math.ceil(pos)-pos)*temp1+(pos-math.floor(pos))*temp2
        rec[x,y] = rec[x,y]+temp/256
        

def proj_forward(sinogram):
    alpha = torch.linspace(0,180,361)*np.pi/180+np.pi/2
    a = torch.linspace(0,183,184)
    b = torch.linspace(-183,-1,183)
    f=torch.cat((a,b))/sinogram.shape[0]
    f = torch.unsqueeze(f, dim=1)                            
    #ramp filter
    fourier_filter = 2 * torch.abs(f)   
    fourier_filter_ = fourier_filter.expand(367,361).unsqueeze(-1)
    fourier_filter_ = torch.transpose(fourier_filter_,0,1)
    fourier_filter_ = torch.cat((fourier_filter_,fourier_filter_),-1)
    
    
    projection = torch.rfft(sinogram, 2, onesided=False).double() * fourier_filter_.double()
    proj_ifft = torch.irfft(projection, 2, onesided=False).float()
    
    proj_ifft = proj_ifft.contiguous()
    
    fbp_host = np.zeros((256,256))
    fbp_dev = cuda.to_device(fbp_host)
    alpha_dev = cuda.to_device(alpha)
    proj_ifft_dev = cuda.to_device(proj_ifft)
    alpha_dev = cuda.to_device(alpha)
    TPB = 16
    threadperblock = (TPB,TPB)
    blockpergrid_x = int(math.ceil(fbp_dev.shape[0]/threadperblock[0]))
    blockpergrid_y = int(math.ceil(fbp_dev.shape[1]/threadperblock[1]))
    blockpergrid = (blockpergrid_x,blockpergrid_y)
    
    func[blockpergrid,threadperblock](alpha_dev,proj_ifft_dev,fbp_dev)
    cuda.synchronize()
    
    im = fbp_dev.copy_to_host()
    im = im/0.06

    return im





views = 361
bins = 367
@cuda.jit
def backfunc(alpha,d_proj,rec):
    x,y = cuda.grid(2)
    for i in range(361):
        theta = alpha[i]
        pos = (x-128)*math.cos(theta)+(y-128)*math.sin(theta)+183
        t1 = math.ceil(pos)
        t2 = math.floor(pos)
        cuda.atomic.max(d_proj, (i,int(t1)), (pos-t2)*rec[x,y]/256)
        cuda.atomic.max(d_proj, (i,int(t2)), (t1-pos)*rec[x,y]/256)
        
def proj_backward(rec):
    alpha = torch.linspace(0,180,361)*np.pi/180+np.pi/2                
    d_proj = np.zeros((views,bins))
    rec_dev = cuda.to_device(rec)
    d_proj_dev = cuda.to_device(d_proj)
    alpha_dev = cuda.to_device(alpha)
    
    TPB = 16
    threadperblock = (TPB,TPB)
    blockpergrid_x = int(math.ceil(rec.shape[0]/threadperblock[0]))
    blockpergrid_y = int(math.ceil(rec.shape[1]/threadperblock[1]))
    blockpergrid = (blockpergrid_x,blockpergrid_y)
    
    backfunc[blockpergrid,threadperblock](alpha_dev,d_proj_dev,rec_dev)
    cuda.synchronize()
    d_p = d_proj_dev.copy_to_host()
    d_p = d_p/0.06

from torch.autograd import Variable 
class myFunction_for_grad(torch.autograd.Function):  
                                                             
    @staticmethod                                  
    def forward(ctx, input_):
#        st = time.time()
        output = proj_forward(input_)
#        et = time.time()
#        print('forward',et-st)  
        return torch.tensor(output)                       
         
    @staticmethod                                 
    def backward(ctx, grad_output):   
#        st = time.time()
        grad = proj_backward(grad_output)
#        print(grad.shape)
#        et = time.time()
#        print('back',et-st)  
        return grad

def RECON(x):
    x = x.cpu()
    l = x.shape[0]
    data1 = x[0]
    data1 = data1.squeeze(0)
    data = myFunction_for_grad.apply(data1)
    data = data.unsqueeze(0)
    data = data.unsqueeze(0)

    for i in range(1,l):
        data_1 = x[i]
        data_1 = data_1.squeeze(0)
        data_ = myFunction_for_grad.apply(data1)
        data_ = data_.unsqueeze(0)
        data_ = data_.unsqueeze(0)
        data = torch.cat((data,data_),dim=0)
    return data
