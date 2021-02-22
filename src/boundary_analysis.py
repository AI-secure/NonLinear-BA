import torch
import argparse
import numpy as np

from tqdm import tqdm

import model_settings
import attack_setting
import constants
import utils
import gradient_generate


def search_boundary(ref_model, src_img, tgt_img, args, threshold=1e-8):
    src_scores = ref_model(src_img)
    src_lbl = src_scores.max(1)[1]
    tgt_scores = ref_model(tgt_img)
    tgt_lbl = tgt_scores.max(1)[1]
    assert src_lbl != tgt_lbl

    lows = torch.zeros(src_img.shape)
    highs = torch.ones(src_img.shape)
    if args.use_gpu:
        lows = lows.cuda()
        highs = highs.cuda()

    while((highs-lows)/threshold).max() > 1:
        mids = (highs+lows) / 2
        mid_img = src_img + mids
        mid_scores = ref_model(mid_img)
        mid_lbl = mid_scores.max(1)[1]

        if mid_lbl == src_lbl:
            lows = mids
        else:
            highs = mids

    return src_img+lows


def prox_omega_one(ref_model, std,
                   p_gen, input_size,
                   src_img, src_lbl, tgt_img, args,
                   n_repeat=100, delta=1e-4, batch_size=32):
    if len(src_img.shape) == 3: # make imgs shape [1, n_channel, height, width]
        src_img = src_img.unsqueeze(0)
        tgt_img = tgt_img.unsqueeze(0)
    print('src_img.shape')
    print(src_img.shape)
    bdr_img = search_boundary(ref_model, src_img, tgt_img, args=args, threshold=1e-8) # binary search for boundary point
    bdr_grad = gradient_generate.calc_gt_grad(ref_model=ref_model, Xs=bdr_img, preprocess_std=std)
    gt_img = bdr_img + bdr_grad/torch.sqrt((bdr_grad**2).sum((1,2,3),keepdim=True))*delta
    gt_lbl = ref_model(gt_img).max(1)[1]
    print('gt_lbl')
    print(gt_lbl)
    assert gt_lbl == src_lbl

    torch_cos_sim_func = torch.nn.CosineSimilarity()

    n_useful = []
    with tqdm(range(int(n_repeat/batch_size)+1)) as pbar:
        for i in pbar:
            B = min(n_repeat, batch_size*(i+1)) - batch_size*i

            latent_data = torch.randn((B, *input_size))
            if p_gen is not None:
                if args.use_gpu:
                    latent_data = latent_data.cuda()
                projected_pert = p_gen.project(latent_data)
                # projected_np = projected_data.detach().cpu().numpy().reshape(B, -1)
            else:
                projected_pert = latent_data
                # projected_np = projected_data.numpy().reshape(B, -1)

            bdr_grad_repeats = bdr_grad.repeat([B, 1, 1, 1])
            cos_sims = torch_cos_sim_func(bdr_grad_repeats.reshape(B, -1), projected_pert.reshape(B, -1))
            cos_sim_signs = (cos_sims > 0)
            print('cos_sims')
            print(cos_sims)
            print('cos_sim_signs')
            print(cos_sim_signs)

            perturb_imgs = bdr_img + projected_pert
            perturb_lbls = ref_model(perturb_imgs).max(1)[1]
            lbl_signs = (perturb_lbls == gt_lbl)
            print('perturb_lbls')
            print(perturb_lbls)
            print('lbl_signs')
            print(lbl_signs)

            print(cos_sim_signs == lbl_signs)
            n_useful += list((cos_sim_signs == lbl_signs).cpu().numpy())
            print('n_useful')
            print(n_useful)
            print(np.sum(n_useful))
            assert 0


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
    # prox_omega(pgen_type=args.pgen, args=args, n_repeat=100)
    mean, std = constants.plot_mean_std(TASK=args.TASK)
    BATCH_SIZE = 32
    model_file_name = model_settings.get_model_file_name(TASK=args.TASK, args=args)
    imgloader = utils.get_imgset(TASK=args.TASK, args=args, is_train=True, BATCH_SIZE=BATCH_SIZE)

    if args.TASK == 'mnist':
        from models import MNISTDNN
        num_class = 10
        img_size = args.mnist_img_size
        ref_model = MNISTDNN(model_type='res18', gpu=args.use_gpu, n_class=num_class, img_size=img_size).eval()
        ref_model.load_state_dict(torch.load('../class_models/%s_%s.model'%(model_file_name, 'res18')))
    elif args.TASK == 'cifar10':
        from models import CifarDNN
        img_size = args.cifar10_img_size
        ref_model = CifarDNN(model_type='res18', pretrained=False, gpu=args.use_gpu, img_size=img_size).eval()
        ref_model.load_state_dict(torch.load('../class_models/%s_res18.model' % (model_file_name)))
    else:
        print("To be implemented")
        assert 0

    p_gen = attack_setting.load_pgen(task=args.TASK, pgen_type=args.pgen, args=args)
    input_size = attack_setting.pgen_input_size(task=args.TASK, pgen_type=args.pgen, args=args)

    n_prox = 0
    for Xs, ys in imgloader:
        if args.use_gpu:
            Xs = Xs.cuda()
            ys = ys.cuda()

        img_scores = ref_model(Xs)
        img_lbls = img_scores.max(1)[1]
        correct_idxs = (img_lbls==ys)
        correct_Xs = Xs[correct_idxs]
        correct_ys = ys[correct_idxs]

        if n_prox > args.N_img:
            break
        else:
            n_c = len(correct_ys)
            for i in range(int(n_c/2)):
                if correct_ys[i] == correct_ys[n_c-i-1]:
                    continue
                n_prox += 1
                prox_omega_one(ref_model, std, p_gen, input_size,
                               src_img=correct_Xs[i], src_lbl=correct_ys[i], tgt_img=correct_Xs[n_c-i-1],
                               args=args,
                                n_repeat=100, delta=1e-4, batch_size=32)