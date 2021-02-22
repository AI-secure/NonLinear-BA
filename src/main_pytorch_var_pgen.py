import foolbox
import numpy as np
from foolbox.criteria import TargetClass
import argparse
import json
from attack_setting import load_pgen, imagenet_attack, celeba_attack
from tqdm import tqdm

def MSE(x1, x2):
    return ((x1-x2)**2).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model_discretize', action='store_true')
    parser.add_argument('--attack_discretize', action='store_true')
    parser.add_argument('--atk_level', type=int, default=999)
    # parser.add_argument('--pgen', type=str)
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--N_img', type=int, default=50)
    args = parser.parse_args()

    TASK = args.TASK
    N_img = args.N_img
    N_repeat = 1

    np.random.seed(0)
    if TASK == 'imagenet':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = imagenet_attack(args, N_img)
        print("Not implemented yet")
        assert False
    elif TASK == 'celeba':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = celeba_attack(args, N_img)
        pgen_list = ['GAN9408', 'PCA9408']
    else:
        raise NotImplementedError()
    print ("Setting loaded")
    print ("Task: %s; Number of Image: %s; Number of repeat: %s"%(TASK, N_img, N_repeat))

    all_logs = []
    with tqdm(range(N_img)) as pbar:
        for i in pbar:
            src_image, src_label, tgt_image, tgt_label = src_images[i], src_labels[i], tgt_images[i], tgt_labels[i]
            attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
            adv = None
            for PGEN in pgen_list:
                p_gen = load_pgen(TASK, PGEN, args)
                if TASK == 'imagenet' or TASK == 'celeba':
                    if PGEN == 'naive':
                        ITER = 200
                        maxN = 100
                        initN = 100
                    elif PGEN.startswith('PCA') or PGEN.startswith('AE') or PGEN.startswith('VAE') or PGEN.startswith(
                            'oldAE'):
                        ITER = 200
                        maxN = 100
                        initN = 100
                    elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                        ITER = 200
                        maxN = 100
                        initN = 100
                    elif PGEN == 'NNGen' or PGEN.startswith('GAN'):
                        ITER = 500
                        maxN = 30
                        initN = 30
                    else:
                        raise NotImplementedError()

                print("PGEN:", PGEN)
                if p_gen is None:
                    rho = 1.0
                else:
                    rvs = p_gen.generate_ps(src_image, 10, level=999)
                    grad_gt = fmodel.gradient_one(src_image, label=src_label)
                    rho = p_gen.calc_rho(grad_gt, src_image).item()
                pbar.set_description("src label: %d; tgt label: %d; rho: %.4f"%(src_label, tgt_label, rho))

                for _ in range(N_repeat):
                    if adv != None:
                        start_image = adv.perturbed
                    else:
                        start_image = src_image
                    adv = attack(tgt_image, tgt_label, starting_point = start_image, iterations=ITER,
                                 stepsize_search='geometric_progression', unpack=False, max_num_evals=maxN,
                                 initial_num_evals=initN, internal_dtype=np.float32, rv_generator = p_gen,
                                 atk_level=args.atk_level, mask=mask, batch_size=16, rho_ref = rho,
                                 log_every_n_steps=10, discretize=args.attack_discretize, verbose=False, plot_adv=False)
                    all_logs.append(attack.logger)

    with open('BAPP_result/attack_multi_%s_%s_%s.log'%(TASK, 'var', args.suffix), 'w') as outf:
        json.dump(all_logs, outf)