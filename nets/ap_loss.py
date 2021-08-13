# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import numpy as np
import torch
import torch.nn as nn


class APLoss (nn.Module):
    """ differentiable AP loss, through quantization.
        
        Input: (N, M)   values in [min, max]  this input is a score map
        label: (N, M)   values in {0, 1}      the label is the ground truth for score map
        
        Returns: list of query AP (for each n in {1..N})
                 Note: typically, you want to minimize 1 - mean(AP)
    """
    def __init__(self, nq=25, min=0, max=1, euc=False):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        self.euc = euc
        gap = max - min
        assert gap > 0
        
        # init quantizer = non-learnable (fixed) convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        a = (nq-1) / gap
        #1st half = lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight.data[:nq] = -a
        q.bias.data[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1)) # b = 1 + a*(min+x)
        #2nd half = lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight.data[nq:] = a
        q.bias.data[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min) # b = 1 - a*(min+x)
        # first and last one are special: just horizontal straight line
        q.weight.data[0] = q.weight.data[-1] = 0
        q.bias.data[0] = q.bias.data[-1] = 1

    def compute_AP(self, x, label):
        N, M = x.shape #if two dimensional, then N=(batch*第一图长*第一图宽), M=(batch*第二图长*第二图宽)
        if self.euc:  # euclidean distance in same range than similarities
            x = 1 - torch.sqrt(2.001 - 2*x)

        # quantize all predictions
        q = self.quantizer(x.unsqueeze(1)) #x.unsqueeze(1)gives you dimension of N,1,M; after Conv1d you will get dimension of N,2*nq,M. 
        q = torch.min(q[:,:self.nq], q[:,self.nq:]).clamp(min=0) # N x Q x M,其实q[:,:self.nq]相当于q[:,:self.nq,:], 最后一个dimension省去了
        #这里torch.min(tensor1,tensor2)相当于tensor1和tensor2 element-wise比较大小，同一个element取两个中的较小的。比如tensor1 [[1,2],[5,6]]; tensor2 [[0,3],[2,5]]那么输出[[0,2],[2,5]]
        #因此这里最后q的dimension就是N,q,M
        
        nbs = q.sum(dim=-1) # number of samples  N x Q = c. nbs dimension is N*q
        rec = (q * label.view(N,1,M).float()).sum(dim=-1) # nb of correct samples = c+ N x Q; * here is element-wise multiplication, label is binary mask of the size N*M
        #q * label.view(N,1,M)会返回q的dimension(N*q*M),加上sum(dim=-1)之后dimension最终是N*q
        
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1)) # precision
        rec /= rec.sum(dim=-1).unsqueeze(1) # norm in [0,1]; rec.sum(dim=-1).unsqueeze(1) dimension is N*1. So it is (N*q)/(N*1), which finally gives you N*q
        #rec is the recall
        
        ap = (prec * rec).sum(dim=-1) # per-image AP; ap dimension is N,this is more of like per-pixel AP loss
        return ap 

    def forward(self, x, label):
        assert x.shape == label.shape # N x M; x is score map, label is ground truth score map
        return self.compute_AP(x, label)





