# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# python cifar10_dcgan.py --dataset cifar10 --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 32 --cuda --outf out_cifar --manualSeed 13 --niter 100

class Cifar10DCGenerator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=100, ngf=64):
        super(Cifar10DCGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Tanh()
            nn.ReLU(True)
        )
        self.out_conv = nn.ConvTranspose2d(nc, nc, kernel_size=7, stride=7, padding=0)
        self.out_nonlin = nn.Tanh()
        # self.gen_fc = nn.Linear(nc*64*64, nc*224*224)
        # self.gen_out = nn.Tanh()

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     o1 = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     o1 = self.main(input)
        # B = o1.shape[0]
        # o2 = self.gen_fc(o1.view(B, -1))
        # output = self.gen_out(o2)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        o = self.out_nonlin(self.out_conv(output))
        # generate 224*224 by default
        # o = torch.nn.functional.interpolate(output, size=(224, 224))
        return o

        # if not test:
        #     # when training, return range [0, 255]
        #     return torch.nn.functional.interpolate(output, size=(224, 224))
        #     # return output
        # else:
        #     # when test, return range [0, 1]
        #     return torch.nn.functional.interpolate(output, size=(224, 224))/255

    def generate_ps(self, inp, N, level=None):
        latent_size = 100
        Z = torch.randn(N, latent_size, 1, 1)
        ps = self.forward(Z).cpu().detach().numpy()
        return ps

    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]

    def project(self, latent_Z):
        ps = self.forward(latent_Z)
        return ps


class Cifar10DCDiscriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Cifar10DCDiscriminator, self).__init__()
        self.ngpu = ngpu
        # self.dis_fc = nn.Linear(nc*224*224, nc*64*64)
        self.in_conv = nn.Conv2d(nc, nc, kernel_size=7, stride=7, padding=0)
        self.in_nonlin = nn.LeakyReLU(0.2, inplace=True)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # B = input.shape[0]
        # i1 = self.dis_fc(input).view(B, -1, 64, 64)
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, i1, range(self.ngpu))
        # else:
        #     output = self.main(i1)

        # linear interpolate gradient imgs
        # input = torch.nn.functional.interpolate(input, size=(32, 32))
        input = self.in_nonlin(self.in_conv(input))

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


