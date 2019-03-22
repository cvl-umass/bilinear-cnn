import torch
import torch.utils.data as data
import numpy as np
from scipy.io import loadmat as loadmat

import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(meta, split, dataset_root, classes, subset=False,
        create_val=True):

    imgList = [str(x[0][0]) for x in meta['annotations'][0]]
    setList = [int(np.squeeze(x[6])) for x in meta['annotations'][0]]
    annoList = [int(np.squeeze(x[5])) for x in meta['annotations'][0]]

    if split == 'train':
        setIdx = [0]
    elif split == 'val':
        setIdx = [1]
    elif split == 'test':
        setIdx = [-1]
    elif split == 'train_val':
        setIdx = [1, 0]
    else:
        ValueError('Unknown split: %s' % split)

    if create_val:
        np.random.seed(0)
        trainList = [idx for idx, v in enumerate(setList) if v==0]
        trainList = np.random.permutation(trainList)
        valNum = np.ceil(0.333*len(trainList)).astype(int)
        valList = trainList[:valNum]
        trainList = trainList[valNum:]

        for idx, v in enumerate(setList):
            if v == 1:
                setList[idx] = -1
        for k in valList:
            setList[k] = 1

        np.random.seed()

    img = []
    count = [0]*len(classes)
    for idx, anno in enumerate(annoList):
        label = anno - 1
        if setList[idx] not in  setIdx:
            continue
        if subset:
            if count[label] >= 5:
                continue
            count[label] += 1
        imageName = os.path.join(dataset_root, imgList[idx])
        img.append((imageName, label))

    return img


class CarsDataset(data.Dataset):
    def __init__(self, dataset_root, split, subset=False, transform=None,
            create_val=True, target_transform=None,
            loader=dataset_parser.default_loader):
        self.loader = loader
        self.dataset_root = dataset_root
        self.imageRoot = os.path.join(dataset_root, 'car_ims')

        meta = loadmat(os.path.join(dataset_root, 'cars_annos.mat'))
        class_meta = meta['class_names'][0]
        self.classes = [np.array_str(x) for x in class_meta]

        self.imgs = make_dataset(meta, split, self.dataset_root,
                        self.classes, subset, create_val=create_val)
        self.transform = transform
        self.target_transform = target_transform
        
        self.dataset_name = 'cars'
        self.load_images = True
        self.feat_root = None

    def __getitem__(self, index):

        if self.load_images:
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = [x(img) for x in self.transform]

            if self.target_transform is not None:
                target = self.target_transform(target)

        else:
            path, target = self.imgs[index]
            path = os.path.join(self.feat_root, path[len(self.imageRoot)+1:-3])
            path = path + 'pt'
            img = torch.load(path)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return (*img, target, path)

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
