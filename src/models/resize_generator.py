import numpy as np
from skimage import transform
import torch

class ResizeGenerator:
    def __init__(self, batch_size=32, factor=4.0, preprocess=None):
        self.batch_size = batch_size
        self.factor = factor
        self.preprocess = preprocess

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        ps = []
        for _ in range(N):
            shape = inp.shape
            assert len(shape)==3 #and shape[0] == 3 # support 1-channel mnist data
            p_small = np.random.randn(shape[0], int(shape[1]/self.factor), int(shape[2]/self.factor))
            #if (_ == 0):
            #    print (p_small.shape)
            p = transform.resize(p_small.transpose(1,2,0), inp.transpose(1,2,0).shape).transpose(2,0,1)
            ps.append(p)
        ps = np.stack(ps, axis=0)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))

        return ps

    def calc_rho(self, gt, inp):
        return np.array([1.0])

    def project(self, latent_Z):
        N = latent_Z.shape[0]
        inp = latent_Z[0].cpu().numpy().copy()

        if self.preprocess is not None:
            transp, mean, std = self.preprocess

        ps = []
        for _ in range(N):
            shape = inp.shape
            assert len(shape) == 3 #and shape[0] == 3 # support 1-channel mnist data
            p_small = np.random.randn(shape[0], int(shape[1] / self.factor), int(shape[2] / self.factor))
            p = transform.resize(p_small.transpose(1, 2, 0), inp.transpose(1, 2, 0).shape).transpose(2, 0, 1)
            ps.append(p)
        ps = np.stack(ps, axis=0)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp + 1))

        ps = torch.FloatTensor(ps)
        return ps.cuda()
