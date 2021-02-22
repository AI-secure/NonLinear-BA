from datasets import CelebAAttributeDataset, CelebAIDDataset
from models import CelebADNN
from models import CelebAIDDNN
from utils import epoch_train, epoch_eval

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import os
import argparse
import constants


def train_celeba_attr_model(attr_o_i='Mouth_Slightly_Open', do_train=False, do_test=True, mounted=False):
    root_dir = f"{constants.ROOT_DATA_PATH}/celeba"

    list_attr_path = '%s/list_attr_celeba.txt' %(root_dir)
    img_data_path = '%s/img_align_celeba' %(root_dir)
    trainset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='train', transform=transform)
    testset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='test', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    if do_train:
        # train celeba attr model
        if not os.path.isdir('../class_models'):
            os.mkdir('../class_models')

        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = CelebADNN(model_type=MODEL_TYPE, gpu=GPU)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            old_acc = 0.0
            worse_acc_cnt = 0
            with tqdm(range(10)) as pbar:
                for _ in pbar:
                    train_loss, train_acc = epoch_train(model, optimizer, trainloader)
                    test_loss, test_acc = epoch_eval(model, testloader)
                    if test_acc > old_acc:
                        torch.save(model.state_dict(), '../class_models/celeba_%s_%s.model'%(attr_o_i, MODEL_TYPE))
                        old_acc = test_acc
                        worse_acc_cnt = 0
                    else:
                        worse_acc_cnt += 1
                        if worse_acc_cnt > 2:
                            print("the test acc is getting worse for 3 consecutive epochs, break")
                            break
                    pbar.set_description("Attr %s, Model %s, Train acc %.4f, Test acc %.4f"%(attr_o_i, MODEL_TYPE, train_acc, test_acc))
        print("Model %s, Train acc %.4f, Test acc %.4f"%(MODEL_TYPE, train_acc, test_acc))

    if do_test:
        # test celeba attr model
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = CelebADNN(model_type=MODEL_TYPE, pretrained=False, gpu=GPU).eval()
            model.load_state_dict(torch.load('./class_models/celeba_%s_%s.model' % (attr_o_i, MODEL_TYPE)))
            test_loss, test_acc = epoch_eval(model, testloader)
            print("Model %s, test acc %f" % (MODEL_TYPE, test_acc))


def train_celeba_id_model(num_class, do_train=False, do_test=True, mounted=False):
    root_dir = f"{constants.ROOT_DATA_PATH}/celeba"
    trainset = CelebAIDDataset(root_dir=root_dir, is_train=True, transform=transform, preprocess=False, random_sample=False, n_id=num_class)
    testset = CelebAIDDataset(root_dir=root_dir, is_train=False, transform=transform, preprocess=False, random_sample=False, n_id=num_class)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    if do_train:
        # train celeba attr model
        if not os.path.isdir('../class_models'):
            os.mkdir('../class_models')

        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = CelebAIDDNN(model_type=MODEL_TYPE, num_class=num_class, pretrained=True, gpu=GPU)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            old_acc = 0.0
            worse_acc_cnt = 0
            with tqdm(range(20)) as pbar:
                for _ in pbar:
                    train_loss, train_acc = epoch_train(model, optimizer, trainloader)
                    test_loss, test_acc = epoch_eval(model, testloader)
                    if test_acc > old_acc:
                        torch.save(model.state_dict(), '../class_models/celeba_id_%s.model'%(MODEL_TYPE))
                        old_acc = test_acc
                        worse_acc_cnt = 0
                    else:
                        worse_acc_cnt += 1
                        if worse_acc_cnt > 2:
                            print("the test acc is getting worse for 3 consecutive epochs, break")
                            break
                    pbar.set_description("CelebA ID, Model %s, Train acc %.4f, Test acc %.4f"%(MODEL_TYPE, train_acc, test_acc))
        print("Model %s, Train acc %.4f, Test acc %.4f"%(MODEL_TYPE, train_acc, test_acc))

    if do_test:
        # test celeba attr model
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = CelebAIDDNN(model_type=MODEL_TYPE, num_class=num_class, pretrained=False, gpu=GPU).eval()
            model.load_state_dict(torch.load('../class_models/celeba_id_%s.model' % (MODEL_TYPE)))
            test_loss, test_acc = epoch_eval(model, testloader)
            print("Model %s, test acc %f" % (MODEL_TYPE, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    args = parser.parse_args()

    GPU = True
    BATCH_SIZE = 64
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_celeba_attr_model(attr_o_i='Mouth_Slightly_Open', do_train=False, do_test=True, mounted=args.mounted)
    # train_celeba_id_model(num_class=args.num_class, do_train=args.do_train, do_test=args.do_test, mounted=args.mounted)
