import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import math
import argparse
import matplotlib.pyplot as plt

import utils

from models import MNISTVAEGenerator

def calc_cos_sim(x1, x2, dim=1):
    cos = (x1*x2).sum(dim) / np.sqrt( (x1**2).sum(dim) * (x2**2).sum(dim))
    return cos


def epoch_train(REF, model, optimizer, BATCH_SIZE=32, mounted=False):
    if mounted:
        data_path = 'ANONYMIZED_DIRECTORY/%s_%s/train_batch_%d.npy'
    else:
        data_path = 'ANONYMIZED_DIRECTORY/%s_%s/train_batch_%d.npy'

    # N_used = 200
    N_used = N_train
    model.train()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_lrecon = 0.0
    cum_lkl = 0.0
    cum_cos = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (TASK, REF, idx))
            X = X / np.sqrt((X ** 2).sum(1, keepdims=True))
            # X = utils.validate(X)
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, n_channels, 224, 224))
            X_enc, X_dec, z_mu, z_std = model(X)
            l, l_recon, l_kl = model.loss(X_dec, X, z_mu, z_std)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            cum_loss += l.item() * B
            cum_lrecon += l_recon.item() * B
            cum_lkl += l_kl.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Cur loss = %.6f; Avg loss/lrecon/lkl = %.6f/%.6f/%.6f; Avg cos = %.4f" % (
            l.item(), cum_loss / tot_num, cum_lrecon / tot_num, cum_lkl / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num, cum_lrecon / tot_num, cum_lkl / tot_num


def epoch_eval(REF, model, BATCH_SIZE=32, mounted=False):
    if mounted:
        data_path = 'ANONYMIZED_DIRECTORY/data/%s_%s/test_batch_%d.npy'
    else:
        data_path = 'ANONYMIZED_DIRECTORY/%s_%s/test_batch_%d.npy'

    # N_used = 50
    N_used = N_test
    model.eval()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_lrecon = 0.0
    cum_lkl = 0.0
    cum_cos = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (TASK, REF, idx))
            X = X / np.sqrt((X ** 2).sum(1, keepdims=True))
            # X = utils.validate(X)
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, n_channels, 224, 224))
            with torch.no_grad():
                X_enc, X_dec, z_mu, z_std = model(X)
                l, l_recon, l_kl = model.loss(X_dec, X, z_mu, z_std)

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            cum_loss += l.item() * B
            cum_lrecon += l_recon.item() * B
            cum_lkl += l_kl.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Avg loss/lrecon/lkl = %.6f/%.6f/%.6f; Avg cos = %.4f" % (
            cum_loss / tot_num, cum_lrecon / tot_num, cum_lkl / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num, cum_lrecon / tot_num, cum_lkl / tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--mounted', action='store_true')
    args = parser.parse_args()

    GPU = True
    # TASK = 'celeba'
    TASK = args.TASK
    if TASK == 'mnist' or TASK == 'cifar10':
        TASK = model_settings.get_model_file_name(TASK, args)

    N_Z = 9408
    print(TASK)

    if TASK.startswith('mnist'):
        n_channels = 1
    else:
        n_channels = 3

    model = VAEGenerator(n_channels=n_channels, gpu=GPU)

    if N_Z == 128:
        ENC_SHAPE = (8, 4, 4)
    elif N_Z == 9408:
        ENC_SHAPE = (48, 14, 14)
    else:
        print("Not implemented")
        assert 0
    fixed_noise = np.random.randn(10, *ENC_SHAPE)
    fixed_noise = fixed_noise / np.sqrt((fixed_noise ** 2).sum((1, 2, 3), keepdims=True))
    fixed_noise = torch.FloatTensor(fixed_noise)
    if GPU:
        fixed_noise = fixed_noise.cuda()

    # print (model)
    # inp = np.ones((8,3,224,224))
    # X_enc, X_dec, z_mu, z_std = model(inp)
    # print (X_enc.shape)
    # print (X_dec.shape)
    # print (z_mu.shape)
    # print (z_std.shape)
    # assert 0

    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    REFs = ['dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet']
    for _ in range(100):
        fake_dec = model.decode(fixed_noise)
        # if _ % 10 == 1:
        #     print(fake_dec.shape)
        #     print(fake_dec[0])

        fig = plt.figure(figsize=(20, 8))
        if n_channels == 3:
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                to_plt = fake_dec[i].detach().cpu().numpy().transpose(1, 2, 0)
                to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
                plt.imshow(to_plt)
        else:
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                x = fake_dec[i].detach().cpu().numpy().reshape(224, 224)
                to_plt = (x - x.min()) / (x.max() - x.min())
                plt.imshow(to_plt, cmap='gray')
        plt.savefig('./plots/%s_gradient_vae%d_eg-%d.pdf' % (TASK, N_Z, _))
        plt.close(fig)

        # for REF in np.random.permutation(REFs):
        # for REF in REFs:
        #     print(REF)
        #     print (epoch_train(REF, model, optimizer))
        #     print (epoch_eval(REF, model))
        #     torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_generator.model' %(TASK, N_Z)) #'vae_generator.model')
        print(epoch_train('rnd', model, optimizer, mounted=args.mounted))
        print(epoch_eval('rnd', model, mounted=args.mounted))
        torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_generator.model' % (TASK, N_Z))

        torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_generator_%d.model' % (TASK, N_Z, _))

    model.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model' % (TASK, N_Z)))
    ps = model.generate_ps(None, 10)
    print(ps.shape)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        to_plt = ps[i].transpose(1, 2, 0)
        to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
        plt.imshow(to_plt)
    plt.savefig('./gen_results/%s_gradient_vae_%d_eg.pdf' % (TASK, N_Z))
