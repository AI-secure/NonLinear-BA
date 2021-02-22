import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import tqdm
from PIL import Image

N_TOT = 25000
N_CAT = int(N_TOT/2)
N_DOG = int(N_TOT/2)
N_NL = 12500 # test set without label


class DogCatDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.file_list = []
        self.label_list = []

        r_train = 0.8
        if self.mode == 'train':
            for i in range(int(N_CAT*r_train)):
                self.file_list.append('train/cat.%d.jpg' %(i))
                self.label_list.append(0)
            for i in range(int(N_DOG*r_train)):
                self.file_list.append('train/dog.%d.jpg' %(i))
                self.label_list.append(1)
        elif self.mode == 'test':
            for i in range(int(N_CAT*r_train), N_CAT):
                self.file_list.append('train/cat.%d.jpg' % (i))
                self.label_list.append(0)
            for i in range(int(N_DOG*r_train), N_DOG):
                self.file_list.append('train/dog.%d.jpg' %(i))
                self.label_list.append(1)
        # elif self.mode == 'grad':
        #     # used for generating gradient images
        #     # TODO: if results are good, change to only using the 'test' set (without labels)
        #     for i in range(N_CAT):
        #         self.file_list.append('train/cat.%d.jpg' % (i))
        #         self.label_list.append(0)
        #     for i in range(N_DOG):
        #         self.file_list.append('train/dog.%d.jpg' %(i))
        #         self.label_list.append(1)
        #     for i in range(N_NL):
        #         self.file_list.append('test/%d.jpg' %(i))
        #         self.label_list.append(2)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # print(self.root_dir)
        # print(os.path.join(self.root_dir, '/', self.file_list[idx]))
        img = Image.open(self.root_dir+'/'+self.file_list[idx])
        if self.transform:
            img = self.transform(img)
        # img = img.numpy()
        return img, self.label_list[idx]


data_transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.ColorJitter(),
    # transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(128),
    transforms.Resize(224),
    transforms.ToTensor()
])
