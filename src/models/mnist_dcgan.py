# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py
# https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/mnist_dcgan/dcgan.py

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

# python mnist_dcgan.py --dataset mnist --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 28 --cuda --outf . --manualSeed 13 --niter 100

class MNISTDCGenerator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(MNISTDCGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input, test=False):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        if not test:
            return output
        else:
            return torch.nn.functional.interpolate(output, size=(224, 224))/255

    def generate_ps(self, inp, N, level=None):
        latent_size = 100
        # Z = torch.FloatTensor(N, self.n_z).normal_()
        Z = torch.randn(N, latent_size, 1, 1)
        ps = self.forward(Z, test=True).cpu().detach().numpy()
        return ps

    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]

    def project(self, latent_Z):
        ps = self.forward(latent_Z, test=True)
        return ps


class MNISTDCDiscriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(MNISTDCDiscriminator, self).__init__()
        self.ngpu = ngpu
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
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


if __name__ == '__main__':
    D = MNISTDCDiscriminator(ngpu=1)
    G = MNISTDCGenerator(ngpu=1)

    D.load_state_dict(torch.load('weights/MNIST_netD_epoch_99.pth'), strict=False)
    G.load_state_dict(torch.load('weights/MNIST_netG_epoch_99.pth'), strict=False)
    D = D.cuda()
    G = G.cuda()

    batch_size = 25
    latent_size = 100

    fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
    if torch.cuda.is_available():
        fixed_noise = fixed_noise.cuda()
    fake_images = G(fixed_noise, test=True)

    # z = torch.randn(batch_size, latent_size).cuda()
    # z = Variable(z)
    # fake_images = G(z)

    fake_images_np = fake_images.cpu().detach().numpy()
    fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 224, 224)
    # fake_images_np = fake_images_np.transpose((0, 2, 3, 1))
    print(fake_images_np[0])
    R, C = 5, 5
    for i in range(batch_size):
        plt.subplot(R, C, i + 1)
        # plt.imshow(fake_images_np[i], interpolation='bilinear')
        plt.imshow(fake_images_np[i], cmap='gray')
    plt.savefig('../gen_results/mnist_dcgan_test_.png')