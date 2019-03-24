import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(dataset_root, split):

    with open(os.path.joint(dataset_root, '%s2018.json'%split)) as f:
        data = json.load(f)
    num_class = len(data['categories'])
    img = [(im['file_name'], annot['category_id']) \
                for im, annot in zip(data['images'], data['annotations'])]

    return img, num_classes


class iNatDataset(data.Dataset):
    def __init__(self, dataset_root, split, transform=None,
            transform_square=None, loader=dataset_parser.default_loader):
        self.loader = loader
        self.dataset_root = dataset_root

        if split == 'train_val':
            self.imgs, self.num_classes = make_dataset(self.dataset_root, 
                    'train')
            self.imgs2, _ = make_dataset(self.dataset_root, 'val')
            self.imgs = self.imgs + self.imgs2
        else:
            self.imgs, self.num_classes = make_dataset(self.dataset_root, 
                        split, self.subtask, from_file=load_from_file)
        self.transform = transform
        self.transform_square = transform_square
        self.dataset_root = dataset_root

    def __getitem__(self, index):
        path, target = self.imgs[index]
        path = os.path.join(self.dataset_root, path)
        img = self.loader(path)
        if self.transform is not None:
            img = [x(img) for x in self.transform]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (*img, target, path)

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return self.num_classes
