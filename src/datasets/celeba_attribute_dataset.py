import numpy as np
import torch
from PIL import Image


# check how imbalance each attribute is
def preprocess_attr_data(list_attr_path):
    attr_pos_dict = {}
    pos_attr_dict = {}
    ys = []
    with open(list_attr_path, 'r') as attr_inf:
        lines = attr_inf.readlines()
        attrs = lines[1].strip().split()
        n_attrs = len(attrs)
        for i in range(n_attrs):
            attr = attrs[i]
            attr_pos_dict[attr] = i
            pos_attr_dict[i] = attr
        n_lines = len(lines)
        for i in range(2, n_lines):
            line_attrs = lines[i].strip().split()
            ys.append([int(line_attrs[0][:-4])] + [int(line_attrs[_]) for _ in range(1, len(line_attrs))])
    ys = np.array(ys)
    diff_ys = np.abs(np.sum(ys[:, 1:], axis=0))
    sorted_diff_ys, sorted_idx = zip(*sorted(zip(diff_ys, np.arange(n_attrs))))
    print(sorted_idx)
    print(sorted_diff_ys)
    print([pos_attr_dict[_] for _ in sorted_idx])
    # ['Attractive', 'Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick', 'High_Cheekbones', 'Male', 'Heavy_Makeup', 'Wavy_Hair', 'Oval_Face', 'Pointy_Nose', 'Arched_Eyebrows', 'Big_Lips', 'Black_Hair', 'Big_Nose', 'Young', 'Straight_Hair', 'Brown_Hair', 'Bags_Under_Eyes', 'Wearing_Earrings', 'No_Beard', 'Bangs', 'Blond_Hair', 'Bushy_Eyebrows', 'Wearing_Necklace', 'Narrow_Eyes', '5_o_Clock_Shadow', 'Receding_Hairline', 'Wearing_Necktie', 'Rosy_Cheeks', 'Eyeglasses', 'Goatee', 'Chubby', 'Sideburns', 'Blurry', 'Wearing_Hat', 'Double_Chin', 'Pale_Skin', 'Gray_Hair', 'Mustache', 'Bald']


class CelebAAttributeDataset(torch.utils.data.Dataset):
    def __init__(self, attr_o_i, list_attr_path, img_data_path, data_split, transform):
        self.attr_o_i = attr_o_i
        self.img_data_path = img_data_path
        self.transform = transform

        with open(list_attr_path, 'r') as attr_inf:
            lines = attr_inf.readlines()
            attrs = lines[1].strip().split()
            n_attrs = len(attrs)
            for i in range(n_attrs):
                attr = attrs[i]
                if attr == self.attr_o_i:
                    self.idx_o_i = i
                    break

        self.img_names = []
        self.ys = []
        n_lines = len(lines)
        # Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
        if data_split == 'all':
            i_range = range(2, n_lines)
        elif data_split == 'train':
            i_range = range(2, 162770 + 2)
        elif data_split == 'val':
            i_range = range(162770 + 2, 182637 + 2)
        elif data_split == 'test':
            i_range = range(182637 + 2, n_lines)
        else:
            print("something wrong here in data_split", data_split)
            assert 0

        for i in i_range:
            line_attrs = lines[i].strip().split()
            self.img_names.append(line_attrs[0])
            self.ys.append(int(line_attrs[1+self.idx_o_i]) == 1)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = '%s/%s' %(self.img_data_path, img_name)
        img = Image.open(img_path)  # .convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.ys[idx])


if __name__ == '__main__':
    preprocess_attr_data('./list_attr_celeba.txt')
