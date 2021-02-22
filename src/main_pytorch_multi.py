import foolbox
import numpy as np
from foolbox.criteria import TargetClass
import argparse
import json
from attack_setting import load_pgen, imagenet_attack, celeba_attack
from attack_setting import celebaid_attack
from attack_setting import cifar10_attack
from attack_setting import mnist_attack
from attack_setting import nclass_attack

from tqdm import tqdm
import nonlinear_analysis
import torch
import model_settings

def MSE(x1, x2):
    return ((x1-x2)**2).mean()

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
    parser.add_argument('--plot_adv', action='store_true')
    parser.add_argument('--celebaid_num_class', type=int, default=10)
    parser.add_argument('--lmbd', default=0.05, type=float)  # for expcos only
    parser.add_argument('--celeba_poi', type=int, default=0)

    parser.add_argument('--smooth', action='store_true') # for VAE generator trained with l1 regularizer
    parser.add_argument('--smooth_suffix', type=str, default='')

    parser.add_argument('--mnist_img_size', type=int, default=28) # resize mnist img
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')
    # parser.add_argument('--mnist_ratio', type=int, default=1) # reduce ratio for mnist10_28
    parser.add_argument('--N_Z', type=int, default=9408)

    parser.add_argument('--cifar10_img_size', type=int, default=32) # resize cifar10
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')

    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    TASK = args.TASK
    # TASK = 'imagenet'
    N_img = args.N_img
    N_repeat = 1
    #PGEN = 'DCT'
    mnist_img_size = args.mnist_img_size

    np.random.seed(0)
    if TASK == 'imagenet':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = imagenet_attack(args, N_img)
    elif TASK == 'cifar10':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = cifar10_attack(args, N_img)
    elif TASK == 'celeba':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = celeba_attack(args, N_img, mounted=args.mounted)
    elif TASK == 'celebaid':
        print("CelebA ID attack. num_class %d" %(args.celebaid_num_class))
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = celebaid_attack(args, N_img, num_class=args.celebaid_num_class, mounted=args.mounted)
    elif TASK == 'mnist':
        print("MNIST10 attack")
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = mnist_attack(args, N_img, num_class=10, mounted=args.mounted)
    elif TASK == 'dogcat2':
        print("Dog-cat classification attack")
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = nclass_attack(args, N_img, task='dogcat', num_class=2, mounted=args.mounted)
    elif TASK == 'celeba2':
        print("CelebA binary classification attack")
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = nclass_attack(args, N_img, task='celeba', num_class=2,mounted=args.mounted)
    else:
        raise NotImplementedError()
    print ("Setting loaded")
    print ("Task: %s; Number of Image: %s; Number of repeat: %s"%(TASK, N_img, N_repeat))

    for PGEN in [args.pgen, ]:
        p_gen = load_pgen(TASK, PGEN, args)
        if TASK == 'imagenet' or TASK == 'celeba' or TASK == 'celebaid' or TASK == 'dogcat2' or TASK == 'celeba2':
            if PGEN == 'naive':
                ITER = 200
                maxN = 100
                initN = 100
            elif PGEN.startswith('PCA') or PGEN.startswith('AE')or PGEN.startswith('VAE') or PGEN.startswith('oldAE') \
                    or PGEN.startswith('expcos'):
                ITER = 200
                maxN = 100
                initN = 100
            elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                ITER = 200
                maxN = 100
                initN = 100
            elif PGEN == 'NNGen' or PGEN.startswith('GAN') or PGEN.startswith('DCGAN'):
                ITER = 500
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()
        elif TASK == 'mnist':
            # ITER = 150
            # maxN = 30
            # initN = 30
            ITER = 200
            maxN = 100
            initN = 100

        elif TASK == 'cifar10':
            # ITER = 150
            # maxN = 30
            # initN = 30
            ITER = 200
            maxN = 100
            initN = 100

        all_logs = []
        print ("PGEN:", PGEN)

        if TASK == 'mnist' or TASK == 'cifar10':
            # model_file_name, output_file_name = model_settings.get_model_file_name(TASK, args)
            model_file_name = model_settings.get_model_file_name(TASK, args)
            # log_file_name = 'BAPP_result/attack_multi_%s_%s_%s.log' % (output_file_name, PGEN, args.suffix)
            log_file_name = 'BAPP_result/attack_multi_%s_%s_%s.log' % (model_file_name, PGEN, args.suffix)
        else:
            log_file_name = 'BAPP_result/attack_multi_%s_%s_%s.log' % (TASK, PGEN, args.suffix)
        print(log_file_name)
        with tqdm(range(N_img)) as pbar:
            for i in pbar:
                src_image, src_label, tgt_image, tgt_label = src_images[i], src_labels[i], tgt_images[i], tgt_labels[i]
                # print (src_image.shape)
                # print (tgt_image.shape)
                # print ("Source Image Label:", src_label)
                # print ("Target Image Label:", tgt_label)
                # print(src_image)

                ### Test generator
                if p_gen is None or args.mode == 'test':
                    rho = 1.0
                else:
                    grad_gt = fmodel.gradient_one(src_image, label=src_label)
                    if PGEN.startswith('AE') or PGEN.startswith('VAE') or PGEN.startswith('oldAE') or PGEN.startswith('GAN'):
                        grad_gt = np.expand_dims(grad_gt, 0)
                        rho = nonlinear_analysis.calc_cosine_sim(task=TASK, pgen_type=PGEN, args=args, gt_dy=grad_gt, n_epoch=10000).item()
                    else:
                        rvs = p_gen.generate_ps(src_image, 10, level=999)
                        rho = p_gen.calc_rho(grad_gt, src_image).item()
                pbar.set_description("src label: %d; tgt label: %d; rho: %.4f"%(src_label, tgt_label, rho))
                # rho = 1
                ### Test generator

                for _ in range(N_repeat):
                    attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
                    adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=ITER,
                                 stepsize_search='geometric_progression', unpack=False, max_num_evals=maxN,
                                 initial_num_evals=initN, internal_dtype=np.float32, rv_generator = p_gen,
                                 atk_level=args.atk_level, mask=mask, batch_size=16, rho_ref = rho,
                                 log_every_n_steps=10, discretize=args.attack_discretize, verbose=False,
                                 suffix='%s_%s_%s'%(TASK, PGEN, args.suffix), plot_adv=args.plot_adv)
                    all_logs.append(attack.logger)

                with open(log_file_name, 'w') as outf:
                    json.dump(all_logs, outf)
        #assert 0
        # if TASK == 'mnist' and args.mnist_img_size != 28:
        #     with open('BAPP_result/attack_multi_%s_%d_%s_%s.log' % (TASK, args.mnist_img_size, PGEN, args.suffix), 'w') as outf:
        #         json.dump(all_logs, outf)
        # elif TASK == 'cifar10' and args.cifar10_img_size != 32:
        #     with open('BAPP_result/attack_multi_%s_%d_%s_%s.log' % (TASK, args.cifar10_img_size, PGEN, args.suffix), 'w') as outf:
        #         json.dump(all_logs, outf)
        # else:
        #     with open('BAPP_result/attack_multi_%s_%s_%s.log'%(TASK, PGEN, args.suffix), 'w') as outf:
        #         json.dump(all_logs, outf)