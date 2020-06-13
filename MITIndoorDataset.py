import torch.utils.data as data
import numpy
import os
from torchvision.datasets import folder as dataset_parser


def make_dataset(
    dataset_root,
    imageRoot,
    split,
    classes,
    class_to_anno,
    subset=False,
    create_val=True,
):

    if split == "test":
        with open(os.path.join(dataset_root, "TestImages.txt"), "r") as f:
            imgList = f.readlines()
    else:
        if split == "val":
            assert create_val
        with open(os.path.join(dataset_root, "TrainImages.txt"), "r") as f:
            imgList = f.readlines()

    imgList = [x.rstrip("\n") for x in imgList]

    if split in ["train", "val"] and create_val:
        valNum = numpy.ceil(0.333 * len(imgList)).astype(int)
        numpy.random.seed(0)
        numpy.random.shuffle(imgList)
        if split == "train":
            imgList = imgList[valNum:]
        else:
            imgList = imgList[:valNum]

        numpy.random.seed()

    annoList = [class_to_anno[x.split("/")[0]] for x in imgList]
    img = []
    for img_name, anno in zip(imgList, annoList):
        img.append((os.path.join(imageRoot, img_name), anno))

    return img


class MITIndoorDataset(data.Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        subset=False,
        transform=None,
        create_val=True,
        target_transform=None,
        loader=dataset_parser.default_loader,
    ):
        self.loader = loader
        self.dataset_root = dataset_root
        self.imageRoot = os.path.join(dataset_root, "Images")
        self.split = split

        self.classes = list(os.listdir(self.imageRoot))
        self.classes.sort()
        self.class_to_anno = {x: i for i, x in enumerate(self.classes)}

        self.imgs = make_dataset(
            self.dataset_root,
            self.imageRoot,
            split,
            self.classes,
            self.class_to_anno,
            subset,
            create_val=create_val,
        )
        self.transform = transform
        self.target_transform = target_transform

        self.dataset_name = "mit_indoor"

    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = [x(img) for x in self.transform]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (*img, target, path)

    def get_num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.imgs)
