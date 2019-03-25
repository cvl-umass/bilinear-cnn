import torch
import torch.utils.data as data
import numpy 
import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(dataset_root, imageRoot, split,  classes, subset=False,
                create_val=True):
    with open(os.path.join(dataset_root, 'train_test_split.txt'), 'r') as f:
        setList = f.readlines()
    with open(os.path.join(dataset_root, 'images.txt'), 'r') as f:
        imgList = f.readlines()
    with open(os.path.join(dataset_root, 'image_class_labels.txt'), 'r') as f:
        annoList = f.readlines()

    if split == 'train':
        setIdx = [1]
    elif split == 'val':
        setIdx = [0]
    elif split == 'test':
        setIdx = [-1]
    elif split == 'train_val':
        setIdx = [1, 0]
    else:
        ValueError('Unknown split: %s' % split)

    setDict = [x.split() for x in setList]
    setDict = {x[0]:int(x[1]) for x in setDict}

    if create_val:
        numpy.random.seed(0)
        trainList = [k for k, v in setDict.items() if v == 1]
        trainList = numpy.random.permutation(trainList)
        valNum = numpy.ceil(0.333*len(trainList)).astype(int)
        valList = trainList[:valNum]
        trainList = trainList[valNum:]

        for k, v in setDict.items():
            if setDict[k] == 0:
                setDict[k] = -1
        for k in valList:
            setDict[k] = 0

        numpy.random.seed()
        
    imgDict = [x.split() for x in imgList]
    imgDict = {x[0]:x[1] for x in imgDict}

    img = []
    count = [0]*len(classes)
    for anno in annoList:
        temp = anno.split()
        label = int(temp[1]) - 1
        imgKey = temp[0]
        if setDict[imgKey] not in setIdx:
            continue
        if subset:
            if count[label] >= 5:
                continue
            count[label] += 1
        imageName = os.path.join(imageRoot, imgDict[imgKey])
        img.append((imageName, label))

    return img

class CUBDataset(data.Dataset):
    def __init__(self, dataset_root, split, subset=False, transform=None,
                create_val=True, target_transform=None,
                loader=dataset_parser.default_loader):
        self.loader = loader
        self.dataset_root = dataset_root
        self.imageRoot = os.path.join(dataset_root, 'images')
        self.split = split

        with open(os.path.join(dataset_root, 'classes.txt'), 'r') as f:
            clsList = f.readlines()

        self.classes = [x.split()[1] for x in clsList]

        self.imgs = make_dataset(self.dataset_root, self.imageRoot, split, 
                                self.classes, subset, create_val=create_val)
        self.transform = transform
        self.target_transform = target_transform

        self.dataset_name = 'cub'
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
