import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(dataset_root, split, subset=None):

    with open(os.path.join(dataset_root, '%s2018.json'%split)) as f:
        data = json.load(f)
    if split != 'test':
        if subset is not None:
            # select the images with the annotations as one of the class in subset
            data['categories'] = [x for x in data['categories'] \
                    if x['supercategory'] == subset]
            subset_cid = [x['id'] for x in data['categories']]
            select_images = [(data['images'][idx], x) \
                    for idx, x in enumerate(data['annotations']) \
                    if x['category_id'] in subset_cid]
            data['images'], data['annotations'] = zip(*select_images)

            # re-index the categories
            cls_mapping = {x['id']: idx \
                        for idx, x in enumerate(data['categories'])}
            for idx, x in enumerate(data['categories']):
                data['categories'][idx]['id'] = cls_mapping[x['id']]
            for idx, x in enumerate(data['annotations']):
                data['annotations'][idx]['category_id'] = \
                            cls_mapping[x['category_id']]

        num_classes = len(data['categories'])
        img = [(im['file_name'], annot['category_id']) \
                    for im, annot in zip(data['images'], data['annotations'])]
        classes = [x['name'] for x in data['categories']]
    else:
        num_classes = -1
        img = [(im['file_name'], -1) for im in data['images']]
        classes = []

    return img, num_classes, classes


class iNatDataset(data.Dataset):
    def __init__(self, dataset_root, split, subset=None, transform=None,
            target_transform=None, loader=dataset_parser.default_loader):

        assert subset in ['Plantae', 'Insecta', 'Aves', \
                'Actinopterygii', 'Fungi', 'Reptilia', 'Mollusca', 'Mammalia', \
                'Animalia', 'Amphibia', 'Arachnida', None]
        self.subset = subset

        self.loader = loader
        self.dataset_root = dataset_root

        if split == 'train_val':
            self.imgs, self.num_classes, self.classes = make_dataset(
                                    self.dataset_root, 'train', subset)
            self.imgs2, _, _ = make_dataset(self.dataset_root, 'val', subset)
            self.imgs = self.imgs + self.imgs2
        else:
            self.imgs, self.num_classes, self.classes = make_dataset(
                                            self.dataset_root, split, subset)
        self.transform = transform
        self.target_transform = target_transform
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
