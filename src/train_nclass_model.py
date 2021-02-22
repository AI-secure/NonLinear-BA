from models import NClassDNN
from datasets import DogCatDataset
from utils import epoch_train, epoch_eval

from datasets import CelebABinIDDataset

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import os
import argparse


def train_nclass_model(trainset, testset, n_class, model_name, do_train=False, do_test=True):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    if do_train:
        if not os.path.isdir('../class_models'):
            os.mkdir('../class_models')

        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = NClassDNN(model_type=MODEL_TYPE, pretrained=True, gpu=GPU, n_class=n_class)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            old_acc = 0.0
            worse_acc_cnt = 0
            with tqdm(range(10)) as pbar:
                for _ in pbar:
                    train_loss, train_acc = epoch_train(model, optimizer, trainloader)
                    test_loss, test_acc = epoch_eval(model, testloader)
                    if test_acc > old_acc:
                        torch.save(model.state_dict(), '../class_models/%s_%s.model'%(model_name, MODEL_TYPE))
                        old_acc = test_acc
                        worse_acc_cnt = 0
                    else:
                        worse_acc_cnt += 1
                        if worse_acc_cnt > 2:
                            print("the test acc is getting worse for 3 consecutive epochs, break")
                            break
                    pbar.set_description("Model name %s, Model %s, Train acc %.4f, Test acc %.4f"
                                         %(model_name, MODEL_TYPE, train_acc, test_acc))
        print("Model %s, Train acc %.4f, Test acc %.4f"%(MODEL_TYPE, train_acc, test_acc))

    if do_test:
        for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
            model = NClassDNN(model_type=MODEL_TYPE, pretrained=False, gpu=GPU, n_class=n_class).eval()
            model.load_state_dict(torch.load('../class_models/%s_%s.model' % (model_name, MODEL_TYPE)))
            test_loss, test_acc = epoch_eval(model, testloader)
            print("Model %s, test acc %f" % (MODEL_TYPE, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--num_class', type=int)
    # the "task" argument does not include the num_class, "TASK" does
    parser.add_argument('--task', type=str)
    parser.add_argument('--celeba_poi', type=int, default=0)
    args = parser.parse_args()

    GPU = True

    BATCH_SIZE = 64
    image_size = 224


    if args.task == 'dogcat':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = DogCatDataset(root_dir='../raw_data/dogcat', mode='train', transform=transform)
        testset = DogCatDataset(root_dir='../raw_data/dogcat', mode='test', transform=transform)
        train_nclass_model(trainset, testset, n_class=args.num_class, model_name='%s%s'%(args.task, args.num_class),
                           do_train=True, do_test=True)

    elif '%s%s'%(args.task, args.num_class) == 'celeba2':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if args.mounted:
            root_dir = 'ANONYMIZED_DIRECTORY/data'
        else:
            root_dir = 'ANONYMIZED_DIRECTORY'
        trainset = CelebABinIDDataset(poi=args.celeba_poi, root_dir=root_dir, is_train=True, transform=transform,
                                      get_data=False, random_sample=False, n_id=10)
        testset = CelebABinIDDataset(poi=args.celeba_poi, root_dir=root_dir, is_train=False, transform=transform,
                                      get_data=False, random_sample=False, n_id=10)
        train_nclass_model(trainset, testset, n_class=args.num_class, model_name='%s%s'%(args.task, args.num_class),
                           do_train=True, do_test=True)


