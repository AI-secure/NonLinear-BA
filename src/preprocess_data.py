import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import math
import argparse
import utils
import model_settings
import constants


def convert_img(in_file_path, out_direct, img_id, _size):
    im = Image.open(in_file_path)
    im = im.resize(_size)
    im.save('%s/%d.jpeg' % (out_direct, img_id))


def preprocess_raw_imagenet():
    _size = (224, 224)
    rootdir = '%s/imagenet/ILSVRC2012/train' %(constants.RAW_DATA_PATH)
    _cap = 300000

    file_name_list = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print os.path.join(subdir, file)
            if file.endswith('JPEG'):
                file_name_list.append(os.path.join(subdir, file))
    id_list = list(range(len(file_name_list)))
    np.random.shuffle(id_list)
    out_direct = '%s/imagenet' %(constants.ROOT_DATA_PATH)
    if not os.path.isdir(out_direct):
        os.mkdir(out_direct)
    for i in range(_cap):
        if i % 10000 == 0:
            print("Convert %d img" % (i))
        convert_img(in_file_path=file_name_list[id_list[i]], out_direct=out_direct, img_id=i, _size=_size)


# N_train and N_test refers to the number of npy files (N_sample = N_ * BATCH_SIZE)
# BATCH_SIZE = 32
def preprocess_rnd_data(TASK, REFs, N_train, N_test):
    path = f'{constants.ROOT_DATA_PATH}/{TASK}_rnd'
    if not os.path.isdir(path):
        os.mkdir(path)

    if TASK == 'celeba':
        # celeba train: 5087, test: 624
        N_train_tot = 5087
        N_test_tot = 624
    elif TASK == 'imagenet':
        # test: 625; train: 8750
        N_train_tot = 8750
        N_test_tot = 625
    elif TASK == 'celebaid':
        N_train_tot = 9
        N_test_tot = 3
    elif TASK.startswith('cifar10'): # including cifar10 and cifar10 resize
        N_train_tot = 1563
        N_test_tot = 313
    elif TASK.startswith('mnist'): # including mnist and mnist resize
        N_train_tot = 1875
        N_test_tot = 313
    elif TASK == 'dogcat2':
        N_train_tot = 625
        N_test_tot = 157
    elif TASK == 'celeba2':
        N_train_tot = 9
        N_test_tot = 3
    else:
        print("Task not implemented")
        assert 0

    cnt_train = 0
    for i in tqdm(range(N_train_tot)):
        N_rand = math.ceil(N_train/N_train_tot)
        rands = np.random.choice(len(REFs), N_rand, replace=False)
        for j in range(N_rand):
            if cnt_train >= N_train:
                break
            REF = REFs[rands[j]]
            validate_data(TASK, REF, 'train', i)
            ori_path = '%s/%s_%s/train_batch_%d.npy' % (constants.ROOT_DATA_PATH, TASK, REF, i)
            lnk_path = '%s/%s_rnd/train_batch_%d.npy' %(constants.ROOT_DATA_PATH, TASK, cnt_train)
            if TASK == 'imagenet':
                cmd = 'mv %s %s' %(ori_path, lnk_path) # only store the rnd set for imagenet
            else:
                cmd = 'ln -s %s %s' %(ori_path, lnk_path)
            os.system(cmd)
            # print(cmd)
            # assert 0
            cnt_train += 1

    cnt_test = 0
    for i in tqdm(range(N_test_tot)):
        N_rand = math.ceil(N_test / N_test_tot)
        rands = np.random.choice(len(REFs), N_rand, replace=False)
        for j in range(N_rand):
            if cnt_test >= N_test:
                break
            REF = REFs[rands[j]]
            validate_data(TASK, REF, 'test', i)
            ori_path = '%s/%s_%s/test_batch_%d.npy' % (constants.ROOT_DATA_PATH, TASK, REF, i)
            lnk_path = '%s/%s_rnd/test_batch_%d.npy' % (constants.ROOT_DATA_PATH, TASK, cnt_test)
            if TASK == 'imagenet':
                cmd = 'mv %s %s' %(ori_path, lnk_path) # only store the rnd set for imagenet
            else:
                cmd = 'ln -s %s %s' % (ori_path, lnk_path)
            os.system(cmd)
            # print(cmd)
            # assert 0
            cnt_test += 1



def validate_data(TASK, REF, mode, idx):
    data_path = '%s/%s_%s/%s_batch_%d.npy'
    X = np.load(data_path%(constants.ROOT_DATA_PATH, TASK, REF, mode, idx))
    X = utils.validate(X)
    np.save(data_path%(constants.ROOT_DATA_PATH, TASK, REF, mode, idx), X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    # parser.add_argument('--N_train', type=int)
    # parser.add_argument('--N_test', type=int)
    parser.add_argument('--mnist_img_size', type=int, default=28)
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--cifar10_img_size', type=int, default=32)
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')

    args = parser.parse_args()

    if args.TASK == 'imagenet' or args.TASK == 'celeba':
        N_train = 15625
        N_test = 625
    elif args.TASK == 'celebaid': # tot 10 person
        N_train = 5*9
        N_test = 5*3
    elif args.TASK == 'cifar10':
        N_train = 5*1563
        N_test = 5*313
    elif args.TASK == 'mnist':
        N_train = 5*1875
        N_test = 5*313
    elif args.TASK == 'dogcat2':
        N_train = 5*625
        N_test = 5*157
    elif args.TASK == 'celeba2':
        N_train = 5*9
        N_test = 5*3
    elif args.TASK == 'imagenet_preprocess':
        preprocess_raw_imagenet()
        assert 0
    else:
        print("TASK not implemented")
        assert 0

    if args.TASK == 'mnist' or args.TASK == 'cifar10':
        # TASK, _ = model_settings.get_model_file_name(args.TASK, args)
        TASK = model_settings.get_model_file_name(args.TASK, args)
    # if args.TASK == 'mnist' and args.mnist_img_size != 28:
    #     TASK = '%s_%d' % (args.TASK, args.mnist_img_size)
    # elif args.TASK == 'cifar10' and args.cifar10_img_size != 32:
    #     TASK = '%s_%d' %(args.TASK, args.cifar10_img_size)
    else:
        TASK = args.TASK

    np.random.seed(0)
    print("rnd set generation for task %s" %(TASK))

    REFs = ['dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet']
    preprocess_rnd_data(TASK=TASK, REFs=REFs, N_train=N_train, N_test=N_test)
