import numpy as np
#import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from models import VAEGenerator
from models import MNIST224VAEGenerator
import utils
import argparse
import matplotlib.pyplot as plt
import model_settings
import constants


# l_alpha = 0.00005
# l_alpha = 1e-8
# l_alpha = 1e-6
# l_alpha = 1e-7
l_alpha = 5e-8


def calc_cos_sim(x1, x2, dim=1):
    cos = (x1*x2).sum(dim) / np.sqrt( (x1**2).sum(dim) * (x2**2).sum(dim))
    return cos

def epoch_train(REF, model, optimizer, BATCH_SIZE = 32, mounted=False):
    if TASK == 'celeba':
        # celeba train: 5087, test: 624
        N_train = 5087
        if REF == 'rnd':
            N_train = 15625
    elif TASK == 'imagenet':
        # test: 625; train: 8750
        N_train = 8750
        if REF == 'rnd':
            N_train = 15625
    elif TASK == 'celebaid' or TASK == 'celeba2':
        N_train = 9
        if REF == 'rnd':
            N_train = 5*9
    elif TASK == 'dogcat2':
        N_train = 625
        if REF == 'rnd':
            N_train = 5*625

    elif TASK.startswith('mnist'):
        N_train = 1875
        if REF == 'rnd':
            N_train = 5*1875
    elif TASK.startswith('cifar10'):
        N_train = 1563
        if REF == 'rnd':
            N_train = 5*1563

    else:
        print("Task not implemented")
        assert 0

    data_path = '%s/%s_%s/train_batch_%d.npy'
    
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
            X = np.load(data_path%(constants.ROOT_DATA_PATH, TASK, REF, idx))
            X = X / np.sqrt((X**2).sum(1, keepdims=True))
            # X = utils.validate(X)
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B,n_channels,224,224))
            X_enc, X_dec, z_mu, z_std = model(X)
            l, l_recon, l_kl = model.loss(X_dec, X, z_mu, z_std)
            if args.smooth:
                l1_crit = torch.nn.L1Loss(reduction='sum')
                reg_loss = 0
                for param in model.parameters():
                    l1_target = torch.from_numpy(np.zeros(param.shape)).float().cuda()
                    reg_loss += l1_crit(param, l1_target)
                l = l + reg_loss*l_alpha
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cos_sim = calc_cos_sim(X.reshape(B,-1), X_dec.cpu().detach().numpy().reshape(B,-1), dim=1)
            cum_loss += l.item() * B
            cum_lrecon += l_recon.item() * B
            cum_lkl += l_kl.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Cur loss = %.6f; Avg loss/lrecon/lkl = %.6f/%.6f/%.6f; Avg cos = %.4f"%(l.item(), cum_loss/tot_num, cum_lrecon/tot_num, cum_lkl/tot_num, cum_cos/tot_num))

    return cum_loss/tot_num, cum_cos/tot_num, cum_lrecon/tot_num, cum_lkl/tot_num

def epoch_eval(REF, model, BATCH_SIZE=32, mounted=False):
    if TASK == 'celeba':
        N_test = 624
        if REF == 'rnd':
            N_test = 625
    elif TASK == 'imagenet':
        N_test = 625
        if REF == 'rnd':
            N_test = 625
    elif TASK == 'celebaid' or TASK == 'celeba2':
        N_test = 3
        if REF == 'rnd':
            N_test = 5*3
    elif TASK == 'dogcat2':
        N_test = 157
        if REF == 'rnd':
            N_test = 5*157

    elif TASK.startswith('mnist'):
        N_test = 313
        if REF == 'rnd':
            N_test = 5*313
    elif TASK.startswith('cifar10'):
        N_test = 313
        if REF == 'rnd':
            N_test = 5*313

    else:
        print("Not implemented")
        assert False

    data_path = '%s/%s_%s/test_batch_%d.npy'

    #N_used = 50
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
            X = np.load(data_path%(constants.ROOT_DATA_PATH, TASK, REF, idx))
            X = X / np.sqrt((X**2).sum(1, keepdims=True))
            # X = utils.validate(X)
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B,n_channels,224,224))
            with torch.no_grad():
                X_enc, X_dec, z_mu, z_std = model(X)
                l, l_recon, l_kl = model.loss(X_dec, X, z_mu, z_std)
                if args.smooth:
                    l1_crit = torch.nn.L1Loss(reduction='sum')
                    reg_loss = 0
                    for param in model.parameters():
                        target = torch.from_numpy(np.zeros(param.shape)).float().cuda()
                        reg_loss += l1_crit(param, target)
                    l = l + reg_loss * l_alpha

            cos_sim = calc_cos_sim(X.reshape(B,-1), X_dec.cpu().detach().numpy().reshape(B,-1), dim=1)
            cum_loss += l.item() * B
            cum_lrecon += l_recon.item() * B
            cum_lkl += l_kl.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Avg loss/lrecon/lkl = %.6f/%.6f/%.6f; Avg cos = %.4f"%(cum_loss/tot_num, cum_lrecon/tot_num, cum_lkl/tot_num, cum_cos/tot_num))

    return cum_loss/tot_num, cum_cos/tot_num, cum_lrecon/tot_num, cum_lkl/tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--N_Z', type=int, default=9408, help='Set to 3136 if MNIST ratio 4')

    parser.add_argument('--mnist_img_size', type=int, default=28)
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--cifar10_img_size', type=int, default=32)
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')

    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--smooth_suffix', type=str, default='')
    args = parser.parse_args()

    GPU = True
    # TASK = 'celeba'
    TASK = args.TASK
    # if TASK == 'mnist' or TASK == 'cifar10':
    #     TASK, output_file_name = model_settings.get_model_file_name(TASK, args)
    # else:
    #     output_file_name = TASK
    if TASK == 'mnist' or TASK == 'cifar10':
        TASK = model_settings.get_model_file_name(TASK, args)

    N_Z = args.N_Z
    print(TASK, N_Z)

    if TASK.startswith('mnist'):
        n_channels = 1
    else:
        n_channels = 3

    if TASK.startswith('mnist') and N_Z == 3136:
        model = MNIST224VAEGenerator(n_channels=n_channels, gpu=GPU)
    else:
        model = VAEGenerator(n_channels=n_channels, gpu=GPU)


    if N_Z == 128:
        ENC_SHAPE = (8, 4, 4)
    elif N_Z == 9408:
        ENC_SHAPE = (48, 14, 14)
    elif N_Z == 3136:
        ENC_SHAPE = (16, 14, 14)
    else:
        print("Not implemented")
        assert 0
    fixed_noise = np.random.randn(10, *ENC_SHAPE)
    fixed_noise = fixed_noise / np.sqrt((fixed_noise ** 2).sum((1, 2, 3), keepdims=True))
    fixed_noise = torch.FloatTensor(fixed_noise)
    if GPU:
        fixed_noise = fixed_noise.cuda()

    #print (model)
    #inp = np.ones((8,3,224,224))
    #X_enc, X_dec, z_mu, z_std = model(inp)
    #print (X_enc.shape)
    #print (X_dec.shape)
    #print (z_mu.shape)
    #print (z_std.shape)
    #assert 0

    
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
        if args.smooth:
            fig_path = './plots/%s_gradient_vae%d_smooth%s_eg-%d.pdf' % (TASK, N_Z, args.smooth_suffix, _)
        else:
            fig_path = './plots/%s_gradient_vae%d_eg-%d.pdf' % (TASK, N_Z, _)
        plt.savefig(fig_path)
        plt.close(fig)

        # for REF in np.random.permutation(REFs):
        # for REF in REFs:
        #     print(REF)
        #     print (epoch_train(REF, model, optimizer))
        #     print (epoch_eval(REF, model))
        #     torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_generator.model' %(TASK, N_Z)) #'vae_generator.model')
        print(epoch_train('rnd', model, optimizer, mounted=args.mounted))
        print(epoch_eval('rnd', model, mounted=args.mounted))
        if args.smooth:
            torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_smooth%s_generator.model' % (TASK, N_Z, args.smooth_suffix))
            torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_smooth%s_generator_%d.model' % (TASK, N_Z, args.smooth_suffix, _))
        else:
            torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_generator.model' %(TASK, N_Z))
            torch.save(model.state_dict(), './gen_models/%s_gradient_vae_%d_generator_%d.model' % (TASK, N_Z, _))

    if args.smooth:
        model.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_smooth%s_generator.model' % (TASK, N_Z, args.smooth_suffix)))
    else:
        model.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model' %(TASK, N_Z)))
    ps = model.generate_ps(None, 10)
    print (ps.shape)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,8))
    for i in range(10):
        plt.subplot(2,5,i+1)
        to_plt = ps[i].transpose(1,2,0)
        to_plt = (to_plt-to_plt.min()) / (to_plt.max()-to_plt.min())
        plt.imshow(to_plt)
    if args.smooth:
        plt.savefig('./gen_results/%s_gradient_vae_%d_smooth%s_eg.pdf'%(TASK, N_Z, args.smooth_suffix))
    else:
        plt.savefig('./gen_results/%s_gradient_vae_%d_eg.pdf' % (TASK, N_Z))
