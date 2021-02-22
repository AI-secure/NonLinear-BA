import numpy as np
import torch
import argparse
import attack_setting
from tqdm import tqdm
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model_settings


# optimize the cosine similarity between the ground truth grad and a vector on the nonlinear subspace
def calc_cosine_sim(task, pgen_type, args, gt_dy, n_epoch=10):
    B = gt_dy.shape[0]
    gt_dy = torch.FloatTensor(gt_dy)
    p_gen = attack_setting.load_pgen(task=task, pgen_type=pgen_type, args=args)
    input_size = attack_setting.pgen_input_size(task=task, pgen_type=pgen_type, args=args)
    latent_data = torch.randn((B, *input_size))
    if args.use_gpu:
        latent_data = latent_data.cuda()
        gt_dy = gt_dy.cuda()
    # Puts the tensor on cuda before setting requires_grad to True
    # Otherwise it would generate "ValueError: can't optimize a non-leaf Tensor"
    latent_data.requires_grad = True

    cos_criterion = torch.nn.CosineSimilarity()
    optimizer = torch.optim.Adam([latent_data], lr=1e-3)

    old_loss = -999
    smallest_loss = 999
    with tqdm(range(n_epoch)) as pbar:
        for i in pbar:
            optimizer.zero_grad()
            projected_data = p_gen.project(latent_data)
            proj_loss = torch.mean(-cos_criterion(projected_data.view(B, -1), gt_dy.view(B, -1)))
            proj_loss.backward()
            old_loss = proj_loss.detach().cpu().numpy()
            if old_loss < smallest_loss:
                smallest_loss = old_loss
            optimizer.step()
            pbar.set_description("Epoch %d, cosine similarity %f" %(i, -old_loss))
    return -smallest_loss


# the internal dimension
def calc_internal_dim(task, pgen_type, args):
    B = 5000
    BATCH_SIZE = 32
    p_gen = attack_setting.load_pgen(task=task, pgen_type=pgen_type, args=args)
    input_size = attack_setting.pgen_input_size(task=task, pgen_type=pgen_type, args=args)
    latent_data = torch.randn((B, *input_size))
    if p_gen is not None:
        if args.use_gpu:
            latent_data = latent_data.cuda()
        # projected_data = p_gen.project(latent_data)
        projected_np = None
        for _i in range(int(B/BATCH_SIZE)+1):
            _data = latent_data[_i*BATCH_SIZE: (_i+1)*BATCH_SIZE]
            _B = _data.shape[0]
            if _B < 1:
                break
            _projected = p_gen.project(_data)
            _np = _projected.detach().cpu().numpy().reshape(_B, -1)
            if projected_np is None:
                projected_np = _np
            else:
                projected_np = np.concatenate((projected_np, _np), axis=0)
    else:
        projected_np = latent_data.numpy().reshape(B, -1)

    model_file_name = model_settings.get_model_file_name(TASK=task, args=args)

    if args.do_svd:
        print('Doing svd...')
        u, s, v = np.linalg.svd(projected_np, full_matrices=False)
        np.save('BAPP_result/%s_%s_internal_dim_u.npy' %(model_file_name, pgen_type), u)
        np.save('BAPP_result/%s_%s_internal_dim_s.npy' % (model_file_name, pgen_type), s)
        np.save('BAPP_result/%s_%s_internal_dim_v.npy' % (model_file_name, pgen_type), v)
    else:
        u = np.load('BAPP_result/%s_%s_internal_dim_u.npy' % (model_file_name, pgen_type))
        s = np.load('BAPP_result/%s_%s_internal_dim_s.npy' % (model_file_name, pgen_type))
        v = np.load('BAPP_result/%s_%s_internal_dim_v.npy' % (model_file_name, pgen_type))
        projected_np = u.dot(np.diag(s)).dot(v)
    cos_sims = []
    s_keep = np.zeros(s.shape)
    with tqdm(range(s.shape[0])) as pbar:
        for i in pbar:
            s_keep [i] = s[i]
            if i % inter_gap == 0:
                slice = u.dot(np.diag(s_keep)).dot(v)
                cos_sim = np.mean(utils.calc_cos_sim(x1=projected_np, x2=slice, dim=1))
                cos_sims.append(cos_sim)
                pbar.set_description('Keep dim %d, cosine similarity %f' %(i, cos_sim))
                np.save('BAPP_result/%s_%s_internal_dim.npy' % (model_file_name, pgen_type), cos_sims)
                if cos_sim > 0.9999:
                    break
    print("%s, %s, Cosine similarity for internal dim done" %(model_file_name, pgen_type))
    print(cos_sims)


def calc_cos_sim(x1, x2, dim=1):
    cos = (x1 * x2).sum(dim) / np.sqrt((x1 ** 2).sum(dim) * (x2 ** 2).sum(dim)+1e-16)
    return cos


# expected cosine similarity
def calc_exp_cos(task, pgen_type, args):
    p_gen = attack_setting.load_pgen(task=task, pgen_type=pgen_type, args=args)
    input_size = attack_setting.pgen_input_size(task=task, pgen_type=pgen_type, args=args)

    REF = 'res18'
    if task == 'mnist':
        img_size = args.mnist_img_size
    elif task == 'cifar10':
        img_size = args.cifar10_img_size
    else:
        img_size = 224

    if args.mounted:
        data_dir = 'ANONYMIZED_DIRECTORY/%s_%s' %(task, REF)
        if task == 'cifar10' and args.cifar10_img_size != 32:
            img_size = args.cifar10_img_size
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' % (task, img_size, REF)
        if task == 'mnist' and args.mnist_img_size != 28:
            img_size = args.mnist_img_size
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' %(task, img_size, REF)
    else:
        data_dir = 'ANONYMIZED_DIRECTORY/%s_%s' %(task, REF)
        if task == 'cifar10' and args.cifar10_img_size != 32:
            img_size = args.cifar10_img_size
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' % (task, img_size, REF)
        if task == 'mnist' and args.mnist_img_size != 28:
            img_size = args.mnist_img_size
            data_dir = 'ANONYMIZED_DIRECTORY/%s_%d_%s' % (task, img_size, REF)

    data_path = data_dir + '/test_batch_%d.npy'

    cos_sims = []

    if task == 'celeba':
        N_test = 624
        if REF == 'rnd':
            N_test = 625
    elif task == 'imagenet':
        N_test = 625
        if REF == 'rnd':
            N_test = 625
    elif task == 'celebaid':
        N_test = 3
        if REF == 'rnd':
            N_test = 5 * 3
    elif task == 'cifar10':
        N_test = 313
        if REF == 'rnd':
            N_test = 5 * 313
    elif args.TASK == 'mnist':
        N_test = 313
        if REF == 'rnd':
            N_test = 5*313

    with tqdm(range(min(10000, N_test))) as pbar:
        for idx in pbar:
            X = np.load(data_path % (idx))

            B = X.shape[0]
            latent_data = torch.randn((B, *input_size))
            if p_gen is not None:
                if args.use_gpu:
                    latent_data = latent_data.cuda()
                projected_data = p_gen.project(latent_data)
                projected_np = projected_data.detach().cpu().numpy().reshape(B, -1)
            else:
                projected_np = latent_data.numpy().reshape(B, -1)
            # if np.sum((X ** 2).sum(1)) < 1e-8 or np.sum((projected_np ** 2).sum(1)) < 1e-8:
            #     continue
            cos_sim = np.mean(np.abs(calc_cos_sim(x1=X, x2=projected_np, dim=1)))
            cos_sims.append(cos_sim)
            pbar.set_description('Data # %d, cosine similarity %f' %(idx, cos_sim))
    print("%s, %s, Expected cos similarity" %(task, pgen_type))
    print(cos_sims)
    np.save('BAPP_result/%s_%d_%s_exp_cos.npy' % (task, img_size, pgen_type), cos_sims)
    return np.mean(cos_sims)


def plot_inter_dim(task, pgens, args):
    if task == 'mnist':
        img_size = args.mnist_img_size
    elif task == 'cifar10':
        img_size = args.cifar10_img_size
    else:
        img_size = 224

    fig = plt.figure(figsize=(10, 10))
    for pgen_type in pgens:
        inter_dim_cos = np.load('BAPP_result/%s_%d_%s_internal_dim.npy'%(task, img_size, pgen_type))
        inter_dim_s = np.load('BAPP_result/%s_%d_%s_internal_dim_s.npy' % (task, img_size, pgen_type))
        xs = []
        for i in range(len(inter_dim_s)):
            if i % inter_gap == 0:
                xs.append(i)
        # assert len(xs) == len(inter_dim_cos)
        xs = xs[:len(inter_dim_cos)]
        plt.plot(xs, inter_dim_cos, label='%s_%d %s' %(task, img_size, pgen_type))
    plt.legend()
    plt.xlabel('# dim kept')
    plt.ylabel('Reconstruction proportion')
    plt.savefig('./plots/%s_%d_internal_dim.png' %(task, img_size), bbox_inches='tight')


def plot_exp_cos(task, pgens, args):
    if task == 'mnist':
        img_size = args.mnist_img_size
    elif task == 'cifar10':
        img_size = args.cifar10_img_size
    else:
        img_size = 224

    cos_means = []
    for pgen_type in pgens:
        exp_cos = np.load('BAPP_result/%s_%d_%s_exp_cos.npy' % (task, img_size, pgen_type))
        mean_cos = np.mean(exp_cos)
        xs = list(range(len(exp_cos)))
        fig = plt.figure(figsize=(10, 10))
        plt.plot(xs, exp_cos, label='%s_%d %s, mean %f' % (task, img_size, pgen_type, mean_cos))
        plt.legend()
        plt.xlabel('# generated grad')
        plt.ylabel('Cos similarity with ground truth grad')
        plt.savefig('./plots/exp_cos_%s_%d_%s.png' % (task, img_size, pgen_type), bbox_inches='tight')
        cos_means.append(mean_cos)
    print(cos_means)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model_discretize', action='store_true')
    parser.add_argument('--attack_discretize', action='store_true')
    parser.add_argument('--atk_level', type=int, default=999)
    parser.add_argument('--pgen', type=str)
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--N_img', type=int, default=50)
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--lmbd', default=0.05, type=float) # for expcos only
    parser.add_argument('--do_svd', action='store_true')

    parser.add_argument('--mnist_img_size', type=int, default=28)  # resize mnist img
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--cifar10_img_size', type=int, default=32)  # resize cifar10
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')
    args = parser.parse_args()

    np.random.seed(0)

    inter_gap = 50

    calc_internal_dim(task=args.TASK, pgen_type=args.pgen, args=args)





