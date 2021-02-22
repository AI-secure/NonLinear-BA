import skimage
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Y' = 0.2989 R + 0.5870 G + 0.1140 B
def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])


def calc_img_entropy(img):
    ent = skimage.measure.shannon_entropy(img)
    return ent


def dataset_img_entropy(TASK, REF):
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
            N_test = 5 * 3
    elif TASK == 'dogcat2':
        N_test = 157
        if REF == 'rnd':
            N_test = 5 * 157
    else:
        print("Not implemented")
        assert False

    img_entropys = []
    # for _i in range(N_test):
    with tqdm(range(N_test)) as pbar:
        for _i in pbar:
            data_path = '%s/%s_%s/test_batch_%d.npy' % (root_dir, TASK, REF, _i)
            X = np.load(data_path)
            for _j in range(X.shape[0]):
                gray_x = rgb2gray(X[_j].reshape(3,224,224).transpose(1, 2, 0))

                # if _i == 0 and _j < 10:
                #     fig = plt.figure()
                #     to_plt = (gray_x - gray_x.min()) / (gray_x.max() - gray_x.min())
                #     plt.imshow(to_plt, cmap='gray')
                #     fig.savefig('./plots/%s_%s_entropy_data_%d_%d.png' % (TASK, REF, _i, _j))
                # else:
                #     assert 0

                print('gaussian 224')
                test_x = np.random.normal(size=[224, 224])
                print(test_x)
                test_entropy = calc_img_entropy(test_x)
                print(test_entropy)

                print('gaussian 28')
                test_x = np.random.normal(size=[28, 28])
                print(test_x)
                test_entropy = calc_img_entropy(test_x)
                print(test_entropy)
                #
                # print('camera')
                # from skimage import data
                # print(calc_img_entropy(data.camera()))
                #
                print('data')
                print(gray_x)
                _entropy = calc_img_entropy(gray_x)
                img_entropys.append(_entropy)
                print(_entropy)
                assert 0
            pbar.set_description('chunk %d' % (_i))
    np.save('%s/%s_%s/entropy.npy' % (root_dir, TASK, REF), img_entropys)


def plot_entropy(TASK, REF):
    img_entropys = np.load('%s/%s_%s/entropy.npy' % (root_dir, TASK, REF))
    print(img_entropys)
    # print(len(img_entropys))
    fig = plt.figure()
    plt.hist(img_entropys, density=True, bins=30)
    # _hist, _bin_edges = np.histogram(img_entropys, bins=20)
    # plt.
    plt.title("%s %s Histogram" % (TASK, REF))
    plt.show()
    fig.savefig('./plots/%s_%s_entropy_histogram.png' % (TASK, REF))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str, default='all')
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--rnd_REF', action='store_true')
    parser.add_argument('--do_calc', action='store_true')
    args = parser.parse_args()

    if args.mounted:
        root_dir = '/home/hcli/data'
    else:
        root_dir = '/data/hcli'

    REFs = ['res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet']

    if args.rnd_REF:
        REF = 'rnd'
        if args.do_calc:
            dataset_img_entropy(TASK=args.TASK, REF=REF)
        plot_entropy(TASK=args.TASK, REF=REF)
    else:
        for REF in REFs:
            if args.do_calc:
                dataset_img_entropy(TASK=args.TASK, REF=REF)
            plot_entropy(TASK=args.TASK, REF=REF)
