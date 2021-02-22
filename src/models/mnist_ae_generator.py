import torch
import torch.nn as nn
import numpy as np


# Input size: [batch, 1, 28, 28]
# Output size: [batch, 1, 28, 28]
class MNISTAEGenerator(nn.Module):
    def __init__(self, gpu, N_Z, preprocess=None):
        super(MNISTAEGenerator, self).__init__()
        self.N_Z = N_Z
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 14*28),
            nn.Tanh(),
            nn.Linear(14*28, 14*14),
            nn.Tanh(),
            nn.Linear(14*14, self.N_Z),
        )
        self.ENC_SHAPE = (self.N_Z, )
        self.decoder = nn.Sequential(
            nn.Linear(self.N_Z, 14*14),
            nn.Tanh(),
            nn.Linear(14*14, 14*28),
            nn.Tanh(),
            nn.Linear(14*28, 28 * 28),
        )
        self.loss_fn = nn.MSELoss()
        self.preprocess = preprocess
        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def encode(self, x):
        x = self.encoder(x)
        x_enc = x / torch.sqrt((x ** 2).sum((1,), keepdim=True))
        return x_enc

    def decode(self, x_enc):
        x = self.decoder(x_enc)
        x = x / torch.sqrt((x ** 2).sum((1,), keepdim=True))
        return x

    # below: same with ae_generator
    def forward(self, x):
        x = torch.FloatTensor(x)
        if self.gpu:
            x = x.cuda()

        x_norm = torch.sqrt((x**2).sum((1,),keepdim=True))
        x = x / x_norm

        x_enc = self.encode(x)
        x = self.decode(x_enc)

        x = x * x_norm
        return x_enc, x

    def loss(self, pred, gt):
        gt_var = torch.FloatTensor(gt)
        if self.gpu:
            gt_var = gt_var.cuda()

        pred = pred / torch.sqrt((pred**2).sum((1,),keepdim=True))
        gt_var = gt_var / torch.sqrt((gt_var**2).sum((1,),keepdim=True))
        return self.loss_fn(pred, gt_var)
        #return 1-self.loss_fn(pred, gt_var).mean()

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv**2).sum((1,),keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        ps_var = self.decode(rv_var)
        ps = ps_var.detach().cpu().numpy().reshape(-1, 1, 28, 28)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp):
        x_enc, x_dec = self.forward(gt[None])
        x_dec = x_dec.squeeze(0).detach().cpu().numpy()
        rho = (gt*x_dec).sum() / np.sqrt( (gt**2).sum() * (x_dec**2).sum() )
        return rho

    def project(self, latent_Z):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess

        ps = self.decode(latent_Z)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps
