import torch
import torch.nn as nn
from torchvision import transforms
import os
from config import dset_root
import argparse
import logging
from BCNN import create_bcnn_model
import sys

pretrain_folder = "pretrained_models"


def initializeLogging(logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    return log


def test_model(model, criterion, dset_loader, logger_name=None):

    if logger_name is not None:
        logger = logging.getLogger(logger_name)
    device = next(model.parameters()).device
    model.eval()

    running_corrects = 0
    for idx, all_fields in enumerate(dset_loader):
        if logger_name is not None and (idx + 1) % 10 == 0:
            logger.info("%d / %d" % (idx + 1, len(dset_loader)))
        labels = all_fields[-2]
        inputs = all_fields[:-2]
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(*inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(dset_loader.dataset)

    if logger_name is not None:
        logger.info("Test accuracy: {:.3f}".format(test_acc))


def main(args):
    model_path = os.path.join(pretrain_folder, args.pretrained_filename)
    input_size = args.input_size

    _ = initializeLogging("mylogger")

    if args.dataset in ["cars", "aircrafts"]:
        keep_aspect = False
    else:
        keep_aspect = True

    if args.dataset in ["aircrafts"]:
        crop_from_size = [(x * 256) // 224 for x in input_size]
    else:
        crop_from_size = input_size

    if not keep_aspect:
        input_size = [(x, x) for x in input_size]
        crop_from_size = [(x, x) for x in crop_from_size]

    data_transforms = [
        transforms.Compose(
            [
                transforms.Resize(x[0]),
                transforms.CenterCrop(x[1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        for x in zip(crop_from_size, input_size)
    ]

    if args.dataset == "cub":
        from CUBDataset import CUBDataset as Dataset
    elif args.dataset == "cars":
        from CarsDataset import CarsDataset as Dataset
    elif args.dataset == "aircrafts":
        from AircraftsDataset import AircraftsDataset as Dataset
    else:
        raise ValueError("Unknown dataset: %s" % args.dataset)

    # TODO: check the split name
    dset_test = Dataset(dset_root[args.dataset], "test", transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(
        dset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_bcnn_model(
        args.model_names_list,
        len(dset_test.classes),
        args.pooling_method,
        False,
        True,
        args.embedding_dim,
        2,
        m_sqrt_iter=args.matrix_sqrt_iter,
        proj_dim=args.proj_dim,
    )
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}')".format(model_path))
    else:
        raise ValueError("pretrained model %s does not exist" % (model_path))

    test_model(model, criterion, test_loader, "mylogger")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="size of mini-batch that can fit into gpus",
    )
    parser.add_argument(
        "--pretrained_filename", type=str, help="file name of pretrained model",
    )
    parser.add_argument(
        "--dataset", default="cub", type=str, help="cub | cars | aircrafts"
    )
    parser.add_argument(
        "--input_size",
        nargs="+",
        default=[448],
        type=int,
        help="input size as a list of sizes",
    )
    parser.add_argument(
        "--model_names_list",
        nargs="+",
        default=["vgg"],
        type=str,
        help="input size as a list of sizes",
    )
    parser.add_argument(
        "--pooling_method",
        default="outer_product",
        type=str,
        help="outer_product | sketch | gamma_demo | sketch_gamma_demo",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=8192,
        help="the dimension for the tnesor sketch approximation",
    )
    parser.add_argument(
        "--matrix_sqrt_iter",
        type=int,
        default=0,
        help="number of iteration for the Newtons Method approximating"
        + "matirx square rooti. Default=0 [no matrix square root]",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=0,
        help="project the dimension of cnn features to lower "
        + "dimensionality before computing tensor product",
    )
    parser.add_argument(
        "--gamma",
        default=0.5,
        type=float,
        help="the value of gamma for gamma democratic aggregation",
    )
    args = parser.parse_args()

    main(args)
