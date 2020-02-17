import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from config import dset_root, setup_dataset
import random
import argparse
import copy
import logging
import sys
import time
import shutil
from BCNN_exp import create_bcnn_model
import matplotlib.pyplot as plt

def test_model(model, dset_loader, logger_name=None, mean_vector=None):

    if logger_name is not None:
        logger = logging.getLogger(logger_name)
    device = next(model.parameters()).device
    model.eval()

    num_classes = dset_loader.dataset.get_num_classes()

    sum_vector = [None] * num_classes
    count = 0
    sum_correlation = [None] * num_classes
    variance_vector = [None] * num_classes
    sum_diff = [None] * num_classes
    count = [0] * num_classes

    for idx, all_fields in enumerate(dset_loader):
        labels = all_fields[-2]
        inputs = all_fields[:-2]
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        l_bs = labels.unique()

        # count += (inputs[0].shape[0] * inputs[0].shape[2] * inputs[0].shape[3])

        with torch.set_grad_enabled(False):
            outputs_, features_ = model(*inputs)
            features_ = features_[0]

            for lb in [x.item() for x in l_bs]:
                select = labels == lb
                features = features_[select,:,:,:]
                count[lb] += (
                        torch.sum(select).item() *
                        inputs[0].shape[2] *
                        inputs[0].shape[3]
                )

                if mean_vector is None and sum_vector[lb] is None:
                    sum_vector[lb] = features.sum(dim=3).sum(dim=2).sum(dim=0)
                elif mean_vector is None:
                    sum_vector[lb] += features.sum(dim=3).sum(dim=2).sum(dim=0)
                else:
                    x = features - mean_vector[lb].view(1,-1,1,1)
                    if sum_diff[lb] is None:
                        sum_diff[lb] = ((x * x).sum(dim=3).sum(dim=2).sum(dim=0))
                    else:
                        sum_diff[lb] += ((x * x).sum(dim=3).sum(dim=2).sum(dim=0))

                    bs, c1, h1, w1 = x.shape
                    x = x.view(bs, c1, h1*w1)
                    correlation = torch.bmm(x, torch.transpose(x, 1, 2)).sum(dim=0)
                    if sum_correlation[lb] is None:
                        sum_correlation[lb] = correlation
                    else:
                        sum_correlation[lb] += correlation

                # _, preds = torch.max(outputs, 1)

        print('%d / %d'%(idx, len(dset_loader)))

    if mean_vector is None:
        mean_vector = [
                sum_v / count_lb for sum_v, count_lb in zip(sum_vector, count)
        ]
        # mean_vector = sum_vector / count
        return mean_vector
    else:
        std_vec = [
                (sum_diff_v / count_v) ** 0.5 
                for sum_diff_v, count_v in zip(sum_diff, count)
        ]
        # std_vec = (sum_diff / count) ** 0.5
        # normalization = std_vec.view(-1, 1) * std_vec.view(1, -1)
        normalization = [
                stdv.view(-1, 1) * stdv.view(1, -1) for stdv in std_vec
        ]
        # mean_correlation = sum_correlation / count
        # mean_correlation = mean_correlation / normalization 
        mean_correlation = [
            sum_c / norm /count_v 
            for sum_c, count_v, norm in zip(sum_correlation, count, normalization)
        ]
        return mean_correlation
        
    # if logger_name is not None:
    #     logger.info(
    #         '{} Loss: {:.4f} Acc: {:.4f}'.format('Test', test_loss, test_acc)
    #     )

def main(args):
    model_folder = '../exp/%s/%s/checkpoints'%(args.dataset, args.exp_dir)
    model_path = os.path.join(model_folder, 'model_best.pth.tar')
    if not os.path.isfile(model_path):
        model_folder = '../exp/%s/%s/init_checkpoints'%(args.dataset, args.exp_dir)
        model_path = os.path.join(model_folder, 'model_best.pth.tar')

    dataset = args.dataset 
    input_size = args.input_size 

    if args.dataset in ['cars', 'aircrafts']:
        keep_aspect = False
    else:
        keep_aspect = True

    if args.dataset in ['aircrafts']:
        crop_from_size = [(x * 256) // 224 for x in input_size]
    else:
        crop_from_size = input_size

    # batch_size = 64
    model_names_list = args.model_names_list 
    pooling_method = args.pooling_method 
    embedding = args.embedding_dim 
    order = 2
    matrix_sqrt_iter = args.matrix_sqrt_iter 
    fc_bottleneck =args.fc_bottleneck 
    demo_agg = False
    
    if 'inat' in args.dataset:
        split = {'train': 'train', 'val': 'val'}
    else:
        split = {'train': 'train_val', 'val': 'test'}

    if not keep_aspect:
        input_size = [(x, x) for x in input_size]
        crop_from_size = [(x, x) for x in crop_from_size]

    # make sure the dataset is ready
    if 'inat' in args.dataset:
        setup_dataset('inat')
    else:
        setup_dataset(args.dataset)

    # ==================  Craete data loader ==================================
    data_transforms = [transforms.Compose([
            transforms.Resize(x[0]),
            transforms.CenterCrop(x[1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
            for x in zip(crop_from_size, input_size)]


    if args.dataset == 'cub':
        from CUBDataset import CUBDataset as dataset
    elif args.dataset == 'cars':
        from CarsDataset import CarsDataset as dataset
    elif args.dataset == 'aircrafts':
        from AircraftsDataset import AircraftsDataset as dataset
    elif 'inat' in args.dataset:
        from iNatDataset import iNatDataset as dataset
        if args.dataset == 'inat':
            subset = None
        else:
            subset = args.dataset[len('inat_'):]
            subset = subset[0].upper() + subset[1:]
    else:
        raise ValueError('Unknown dataset: %s' % task)

    if 'inat' in args.dataset:
        dset = {x: dataset(dset_root['inat'], split[x], subset, \
                        transform=data_transforms) for x in ['train', 'val']}
        dset_test = dataset(dset_root['inat'], 'val', subset, \
                        transform=data_transforms)
    else:
        dset = {x: dataset(dset_root[args.dataset], split[x], \
                        transform=data_transforms) for x in ['train', 'val']}
        dset_test = dataset(dset_root[args.dataset], 'test', \
                        transform=data_transforms)

    dset_loader = {x: torch.utils.data.DataLoader(dset[x],
                batch_size=args.batch_size, shuffle=False, num_workers=12,
                drop_last=drop_last) \
                for x, drop_last in zip(['train', 'val'], [True, False])}

    # test_loader = torch.utils.data.DataLoader(dset_test,
    #                     batch_size=batch_size, shuffle=False,
    #                     num_workers=8, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_bcnn_model(model_names_list, len(dset_test.classes), 
                    pooling_method, False, True, embedding, order,
                    m_sqrt_iter=matrix_sqrt_iter,
                    fc_bottleneck=fc_bottleneck, proj_dim=args.proj_dim)

    model = model.to(device)
    model = torch.nn.DataParallel(model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}')".format(model_path))

    mean_vector = test_model(model, dset_loader['train'], None, None)
    correlation = test_model(model, dset_loader['train'], None, mean_vector)

    # abs_correlation = torch.abs(correlation)
    abs_correlation = [torch.abs(corr) for corr in correlation]

    # row_sum = abs_correlation.sum(dim=1)
    row_sum = [abs_corr.sum(dim=1) for abs_corr in abs_correlation]

    # row_off_diag = 1 - 1 / row_sum
    row_off_diag = [1 - 1 / rs for rs in row_sum]

    # mean per-category per-row off-diagonal correlation ratio
    mean_row_off_diag = torch.mean(torch.cat(row_off_diag))
    # mean_row_off_diag = row_off_diag.mean()

    # get the histogram for per-category off-diagonal values
    indices = correlation[0].triu(1).nonzero().transpose(1, 0)
    off_diag = torch.cat([corr[indices[0], indices[1]] for corr in correlation])
    mean_off_diag = torch.abs(off_diag).mean()

    plt.hist(off_diag.cpu().detach().numpy(), bins=200, density=True)
    model_saved_path = os.path.dirname(model_path)
    plt.savefig(os.path.join(model_saved_path, 'dist_corr'))
    plt.close()

    # aggregate the correlation for all catrogries into [C x dim x dim] for saving
    correlation = torch.cat(
            [torch.unsqueeze(x, dim=0) for x in correlation],
            dim=0
    ) 
    np.save(
        os.path.join(model_saved_path, 'correlation'),
        correlation.cpu().detach().numpy()
    )


    if not os.path.isdir(os.path.join(model_saved_path, 'correlation_fig')):
        os.makedirs(os.path.join(model_saved_path, 'correlation_fig'))

    for idx, abs_corr in enumerate(abs_correlation):
        plt.imshow(abs_corr.cpu().detach().numpy(), vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(
            os.path.join(
                model_saved_path,
                'correlation_fig',
                'correlation_fig_%d'%(idx + 1)
            )
        )
        plt.close()

    output_text = 'mean row ratio of off diagonal to the diagonal: %.4f\n'%mean_row_off_diag.item()
    output_text += 'mean off diagonal: %.4f\n'%mean_off_diag.item()
    with open(os.path.join(model_saved_path, 'log_correlation.txt'), 'w') as f:
        f.write(output_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int,
            help='size of mini-batch that can fit into gpus (sub bacth size')
    parser.add_argument('--dataset', default='cub', type=str,
            help='cub | cars | aircrafts')
    parser.add_argument('--exp_dir', default='exp', type=str,
            help='foldername where to save the results for the experiment')
    parser.add_argument('--input_size', nargs='+', default=[448], type=int,
            help='input size as a list of sizes')
    parser.add_argument('--model_names_list', nargs='+', default=['vgg'],
            type=str, help='input size as a list of sizes')
    parser.add_argument('--pooling_method', default='outer_product', type=str,
            help='outer_product | sketch | gamma_demo | sketch_gamma_demo')
    parser.add_argument('--embedding_dim', type=int, default=8192,
            help='the dimension for the tnesor sketch approximation')
    parser.add_argument('--matrix_sqrt_iter', type=int, default=0,
            help='number of iteration for the Newtons Method approximating' + \
                    'matirx square root. Default=0 [no matrix square root]')
    parser.add_argument('--fc_bottleneck', action='store_true',
            help='add bottelneck to the fc layers')
    parser.add_argument('--proj_dim', type=int, default=0,
            help='project the dimension of cnn features to lower ' + \
                    'dimensionality before computing tensor product')
    args = parser.parse_args()
    main(args)
