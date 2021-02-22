import numpy as np
from models import PCAGenerator
import utils
from disk_mat import DiskMatrix, ConcatDiskMat

import os
import argparse
import constants

import model_settings


def load_all_grads(task, ref, train=True, N_used=None, mounted=False):
    grads = []
    path = '%s/%s_%s/%s_batch'%(constants.ROOT_DATA_PATH, task, ref, 'train' if train else 'test')
    i = 0
    used_num = 0
    while used_num < N_used and os.path.exists(path+'_%d.npy'%i):
        cur_block = np.load(path+'_%d.npy'%i)
        if used_num + cur_block.shape[0] > N_used:
            cur_block = cur_block[:N_used-used_num]
        used_num += cur_block.shape[0]
        i += 1

        # regularize gradient block
        # cur_block = utils.regularize(cur_block)

        grads.append(cur_block)
    return np.concatenate(grads, axis=0)

def load_diskmat(task, ref, train=True, N_used=None, N_multi=1, mounted=False):
    path = '%s/%s_%s/%s_batch'%(constants.ROOT_DATA_PATH, task, ref, 'train' if train else 'test')
    return DiskMatrix(path, N_used=N_used, N_multi=N_multi)

def norm(A):
    if isinstance(A, np.ndarray):
        return np.linalg.norm(A)
    else:
        return A.norm()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--TASK', type=str)

    parser.add_argument('--mnist_img_size', type=int, default=28)  # resize mnist img
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')
    parser.add_argument('--N_Z', type=int, default=9408)

    parser.add_argument('--cifar10_img_size', type=int, default=32)  # resize cifar10
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')

    args = parser.parse_args()

    GPU = True
    TASK = args.TASK
    model_file_name = model_settings.get_model_file_name(TASK, args)
    N_b = args.N_Z

    REFs = ['dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet']

    TRANSF = 'res18'
    print ("TRANSF:", TRANSF)

    if TASK == 'imagenet' or TASK == 'celeba':
        X_shape = (3,224,224)
        approx = True
        # N_train = 15624 * 32
        # N_test = 624 * 32
    elif TASK.startswith('mnist'):
        img_size = args.mnist_img_size
        X_shape = (1, img_size, img_size)
        if img_size > 28:
            approx = True
        else:
            approx = False
        # N_train = 5*1875 * 32
        # N_test = 5*313 * 32
    elif TASK.startswith('cifar10'):
        img_size = args.cifar10_img_size
        X_shape = (3, img_size, img_size)
        if img_size > 32:
            approx = True
        else:
            approx = False
        # N_train = 5*1563 * 32
        # N_test = 5*313 * 32
    else:
        print("Task not implemented")
        assert 0

    N_train = 5*1500*32
    N_test = 500*32

    save_path = './gen_models/pca_gen_%s_%d.npy'%(model_file_name, N_b)
    print ("save path:", save_path)

    grads_train = load_diskmat(model_file_name, 'rnd', train=True, N_used=N_train, N_multi=50, mounted=args.mounted)
    grads_test = load_all_grads(model_file_name, 'rnd', train=False, N_used=N_test, mounted=args.mounted)
    grads_test_transfer = load_all_grads(model_file_name, TRANSF, train=False, N_used=4000, mounted=args.mounted)

    print (grads_train.shape)
    print (grads_test.shape)
    print (grads_test_transfer.shape)

    model = PCAGenerator(N_b=N_b, X_shape=X_shape, approx=approx)
    model.fit(grads_train)
    model.save(save_path)
    print ("Model Saved")

    approx_test = grads_test.dot(model.basis.transpose())
    approx_test_transfer = grads_test_transfer.dot(model.basis.transpose())
    print("Rho: ?\t%.6f\t%.6f" %(
        norm(approx_test) / norm(grads_test),
        norm(approx_test_transfer) / norm(grads_test_transfer),
        ))
    approx_train = grads_train.dot(model.basis.transpose())
    print (norm(approx_train) / norm(grads_train))
