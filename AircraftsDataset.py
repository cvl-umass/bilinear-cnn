import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(dataset_root, imageRoot, split, level='variant', 
                subset=False):
    if level == 'variant':
        class_meta = 'variants'
    elif level == 'manufacturer':
        class_meta = 'manufacturers'
    elif level == 'family':
        class_meta = 'families'

    if split == 'train':
        split_suffix = '_train'
    elif split == 'val':
        split_suffix = '_val'
    elif split == 'train_val':
        split_suffix = '_trainval'
    elif split == 'test':
        split_suffix = '_test'
    else:
        ValueError('Unknown split: %s' % split)

    with open(os.path.join(dataset_root, 'data', 
        'images_' + level + split_suffix + '.txt')) as f:
        imgAnnoList = f.readlines()
    with open(os.path.join(dataset_root, 'data', class_meta + '.txt')) as f:
        classes = f.readlines()

    class_dict = {x.rstrip():idx for idx, x in enumerate(classes)}

    count = [0]*len(classes)
    img = []
    for x in imgAnnoList:
        imgName = os.path.join(dataset_root, 'data', 'images', x.split()[0]+'.jpg')
        anno = class_dict[x.rstrip().split(' ', 1)[1]]
        if subset:
            if count[anno] >= 5:
                continue
            count[anno] += 1
        img.append((imgName, anno))

    return img, classes

class AircraftsDataset(data.Dataset):
    def __init__(self, dataset_root, split, subset=False, level='variant',
            transform=None, target_transform=None, 
            loader=dataset_parser.default_loader):
        self.loader = loader
        self.dataset_root = dataset_root
        self.imageRoot = os.path.join(dataset_root, 'data', 'images')

        self.imgs, self.classes = make_dataset(self.dataset_root,
                        self.imageRoot, split, level, subset)

        self.transform = transform
        self.target_transform = target_transform

        self.dataset_name = 'aircrafts'
        self.load_images = True
        self.feat_root = None

    def __getitem__(self, index):

        if self.load_images:
            path, target = self.imgs[index]
            img_original = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        else:
            path, target = self.imgs[index]
            path = os.path.join(self.feat_root, path[len(self.imageRoot)+1:-3])
            path = path + 'pt'
            img = torch.load(path)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target, path 
         
    def get_num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.imgs)

    def set_to_load_features(self, feat_root):
        self.load_images = False
        self.feat_root = feat_root

    def set_to_load_images(self):
        self.load_images = True
        self.feat_root = None
