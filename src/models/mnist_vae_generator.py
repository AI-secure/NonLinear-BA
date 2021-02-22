import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vae_generator import double_conv, inconv, down, up, outconv


class MNISTVAEGenerator(nn.Module):
    def __init__(self, latent_Z, gpu=False):
        super(MNISTVAEGenerator, self).__init__()
        self.latent_Z = latent_Z
        self.ENC_SHAPE = (self.latent_Z, )
        self.hidden_d1 = 256
        self.down1 = nn.Linear(784, self.hidden_d1)
        self.down21 = nn.Linear(self.hidden_d1, self.latent_Z)
        self.down22 = nn.Linear(self.hidden_d1, self.latent_Z)
        self.up1 = nn.Linear(self.latent_Z, self.hidden_d1)
        self.up2 = nn.Linear(self.hidden_d1, 784)

    def encode(self, x):
        h1 = nn.LeakyReLU(self.down1(x))
        z_mu = self.down21(h1)
        logvar = self.down22(h1)
        z_std = torch.exp(logvar / 2)
        rv = np.random.randn(z_std.shape)
        rv = rv / np.sqrt((rv ** 2).sum((1, 2, 3), keepdims=True))
        rv_var = torch.FloatTensor(rv)
        x_enc = z_mu + rv_var * z_std
        return z_mu, z_std, x_enc

    def decode(self, x_enc):
        h3 = nn.LeakyReLU(self.up1(x_enc))
        h4 = self.up2(h3)
        h5 = h4 / torch.sqrt((h4 ** 2).sum((1, 2, 3), keepdim=True))
        return h5

    def forward(self, x):
        mu, std, x_enc = self.encode(x)
        x_dec = self.decode(x_enc)
        return x_enc, x_dec, mu, std

    def loss(self, pred, gt, z_mu, z_std):
        gt_var = torch.FloatTensor(gt)
        if self.gpu:
            gt_var = gt_var.cuda()

        pred = pred / torch.sqrt((pred**2).sum((1,2,3),keepdim=True))
        gt_var = gt_var / torch.sqrt((gt_var**2).sum((1,2,3),keepdim=True))
        l_recon = self.loss_fn(pred, gt_var)
        l_kl = (-0.5 * (self.latent_dim + (torch.log(z_std**2)-z_mu**2-z_std**2).sum((1,2,3)))).mean()
        loss = l_recon + self.lam * l_kl

        return loss, l_recon, l_kl

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv**2).sum((1,2,3),keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        ps_var = self.decode(rv_var)
        ps = ps_var.detach().cpu().numpy()

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]

    def project(self, latent_Z):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess

        ps = self.decode(latent_Z)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp + 1))
        return ps




class MNIST224VAEGenerator(nn.Module):
    def __init__(self, n_channels, lam=1e-11, preprocess=None, gpu=False):
        super(MNIST224VAEGenerator, self).__init__()
        self.gpu = gpu
        self.preprocess = preprocess
        self.lam = lam

        self.inc = inconv(n_channels, 24)
        self.down1 = down(24, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 16)
        self.down4_mu = down(16, 16)
        self.down4_std = down(16, 16)
        self.ENC_SHAPE = (16,14,14)
        self.latent_dim = 3136
        #self.v_to_enc = nn.Linear(self.N, C*H*W)
        #self.up1 = up(128+self.ENC_SHAPE[0], 64)
        self.up1 = up(16, 48)
        self.up2 = up(48, 48)
        self.up3 = up(48, 24)
        self.up4 = up(24, 24)
        self.outc = outconv(24, n_channels)
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = nn.CosineSimilarity()
        #self.outc1 = outconv(64, n_channels)
        #self.outc2 = outconv(64, n_channels)
        if self.gpu:
            self.cuda()

    def encode(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        z_mu = self.down4_mu(x)
        z_logstd2 = self.down4_std(x)
        z_std = torch.exp(z_logstd2/2)
        z = torch.zeros_like(z_mu).normal_() * z_std + z_mu
        x_enc = z / torch.sqrt((z**2).sum((1,2,3),keepdim=True))
        return z_mu, z_std, x_enc

    def decode(self, x_enc):
        x = self.up1(x_enc)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        x = x / torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        return x

    def forward(self, x):
        x = torch.FloatTensor(x)
        if self.gpu:
            x = x.cuda()

        x_norm = torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        x = x / x_norm

        z_mu, z_std, x_enc = self.encode(x)
        x = self.decode(x_enc)

        x = x * x_norm
        return x_enc, x, z_mu, z_std

    def loss(self, pred, gt, z_mu, z_std):
        gt_var = torch.FloatTensor(gt)
        if self.gpu:
            gt_var = gt_var.cuda()

        pred = pred / torch.sqrt((pred**2).sum((1,2,3),keepdim=True))
        gt_var = gt_var / torch.sqrt((gt_var**2).sum((1,2,3),keepdim=True))
        l_recon = self.loss_fn(pred, gt_var)
        l_kl = (-0.5 * (self.latent_dim + (torch.log(z_std**2)-z_mu**2-z_std**2).sum((1,2,3)))).mean()
        loss = l_recon + self.lam * l_kl

        return loss, l_recon, l_kl
        #return 1-self.loss_fn(pred, gt_var).mean()

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv**2).sum((1,2,3),keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        ps_var = self.decode(rv_var)
        ps = ps_var.detach().cpu().numpy()

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]

    def project(self, latent_Z):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess

        ps = self.decode(latent_Z)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp + 1))
        return ps



