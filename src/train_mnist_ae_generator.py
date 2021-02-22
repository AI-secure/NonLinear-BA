import numpy as np
# import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from models import MNISTAEGenerator
import math
import utils
import argparse
import matplotlib.pyplot as plt
# from models import MNISTResizeAEGenerator
from models import AEGenerator


def calc_cos_sim(x1, x2, dim=1):
    cos = (x1 * x2).sum(dim) / np.sqrt((x1 ** 2).sum(dim) * (x2 ** 2).sum(dim))
    return cos


def epoch_train(REF, model, optimizer, BATCH_SIZE=32, mounted=False):
    if mounted:
        data_dir = 'ANONYMIZED_DIRECTORY/%s_%s' %(TASK, REF)
        if img_size != 28:
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' %(TASK, img_size, REF)
    else:
        data_dir = 'ANONYMIZED_DIRECTORY/%s_%s' %(TASK, REF)
        if img_size != 28:
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' % (TASK, img_size, REF)

    # N_used = 200
    N_used = N_train
    model.train()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_cos = 0.0
    data_path = data_dir + '/train_batch_%d.npy'
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (idx))
            X = X / np.sqrt((X ** 2).sum(1, keepdims=True))

            B = X.shape[0]
            assert B <= BATCH_SIZE
            # X = np.reshape(X, (B, 1, 28, 28))
            if img_size == 28:
                X = np.reshape(X, (B, img_size * img_size))
            else:
                X = np.reshape(X, (B, 1, img_size, img_size))
            X_enc, X_dec = model(X)
            l = model.loss(X_dec, X)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            if math.isnan(cos_sim.any()):
                print(X)
                print(X_dec)
                assert 0
            if math.isnan(l.item()) or math.isnan(cos_sim.sum()) or math.isnan(cum_cos) or math.isnan(cum_loss):
                print("l", l.item())
                print("cos_sim", cos_sim.sum())
                print("cum_loss", cum_loss)
                print("cum_cos", cum_cos)
                print("X", X)
                print("X_dec", X_dec)
                assert 0
            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description(
                "Cur loss = %.6f; Avg loss = %.6f; Avg cos = %.4f" % (l.item(), cum_loss / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num


def epoch_eval(REF, model, BATCH_SIZE=32, mounted=False):
    if mounted:
        data_dir = 'ANONYMIZED_DIRECTORY/%s_%s' %(TASK, REF)
        if img_size != 28:
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' %(TASK, img_size, REF)
    else:
        data_dir = 'ANONYMIZED_DIRECTORY/%s_%s' %(TASK, REF)
        if img_size != 28:
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' % (TASK, img_size, REF)

    # N_used = 50
    N_used = N_test
    model.eval()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_cos = 0.0
    data_path = data_dir + '/test_batch_%d.npy'
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (idx))
            X = X / np.sqrt((X ** 2).sum(1, keepdims=True))

            # X = utils.validate(X)
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            # X = np.reshape(X, (B, 1, 28, 28))
            if img_size == 28:
                X = np.reshape(X, (B, img_size * img_size))
            else:
                X = np.reshape(X, (B, 1, img_size, img_size))
            with torch.no_grad():
                X_enc, X_dec = model(X)
                l = model.loss(X_dec, X)

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B

            # print()
            # print("l", l.item())
            # print("cos_sim", cos_sim.sum())
            # print("cum_loss", cum_loss)
            # print("cum_cos", cum_cos)
            # print("X", X)
            # print("X_dec", X_dec)
            pbar.set_description("Avg loss = %.6f; Avg cos = %.4f" % (cum_loss / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--N_Z', type=int, default=9408) # for resize generator
    parser.add_argument('--ratio', type=int, default=4) # for mnist with image_size 28
    args = parser.parse_args()

    GPU = True
    TASK = 'mnist'
    img_size = args.img_size

    if img_size == 28:
        N_Z = int(28*28/args.ratio)
        model = MNISTAEGenerator(gpu=GPU, N_Z=N_Z)
        ENC_SHAPE = (N_Z,)
    elif img_size == 224:
        N_Z = args.N_Z
        model = AEGenerator(n_channels=1, gpu=GPU, N_Z=N_Z)
        if N_Z == 128:
            ENC_SHAPE = (8, 4, 4)
        elif N_Z == 9408:
            ENC_SHAPE = (48, 14, 14)
        else:
            print("Not implemented")
            assert 0
    else:
        print("Not implemented")
        assert 0

    N_train = 5 * 1875
    N_test = 5 * 313

    fixed_noise = np.random.randn(10, *ENC_SHAPE)
    fixed_noise = fixed_noise / np.sqrt((fixed_noise ** 2).sum((1,), keepdims=True))
    fixed_noise = torch.FloatTensor(fixed_noise)
    if GPU:
        fixed_noise = fixed_noise.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    
    for _ in range(100):
        fake_dec = model.decode(fixed_noise)
        if img_size == 28:
            assert fake_dec.shape[-1] == img_size*img_size
        else:
            assert fake_dec.shape[-1] == img_size

        fig = plt.figure(figsize=(20, 8))
        # for i in range(10):
        #     plt.subplot(2, 5, i + 1)
        #     to_plt = fake_dec[i].detach().cpu().numpy().transpose(1, 2, 0)
        #     to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
        #     plt.imshow(to_plt)
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            x = fake_dec[i].detach().cpu().numpy().reshape(img_size, img_size)
            to_plt = (x - x.min()) / (x.max() - x.min())
            plt.imshow(to_plt, cmap='gray')
        plt.savefig('./plots/%s_%d_gradient_ae%d_eg-%d.pdf' % (TASK, img_size, N_Z, _))
        plt.close(fig)

        print(epoch_train('rnd', model, optimizer, mounted=args.mounted))
        print(epoch_eval('rnd', model, mounted=args.mounted))

        torch.save(model.state_dict(), './gen_models/%s_%d_gradient_ae_%d_generator.model' % (TASK, img_size, N_Z))

        torch.save(model.state_dict(), './gen_models/%s_%d_gradient_ae_%d_generator_%d.model' % (TASK, img_size, N_Z, _))