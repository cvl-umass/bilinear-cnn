import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from config import dset_root
import random
import argparse
import copy
import logging
import sys
import time
import shutil
from BCNN import create_bcnn_model

def test_model(model, criterion,  dset_loader, logger_name=None):

    if logger_name is not None:
        logger = logging.getLogger(logger_name)
    device = next(model.parameters()).device
    model.eval()

    running_loss = 0.0; running_corrects = 0
    '''
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    ta = time.perf_counter()
    '''
    for idx, all_fields in enumerate(dset_loader):
        labels = all_fields[-2]
        inputs = all_fields[:-2]
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            '''
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            ta = time.perf_counter()
            '''
            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            '''
            torch.cuda.synchronize()
            tb = time.perf_counter()
            print('time: {:.02e}s'.format((tb - ta) / 64))
            '''
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs[0].size(0)
        running_corrects += torch.sum(preds == labels.data)

        print('%d / %d'%(idx, len(dset_loader)))
        
        '''
        if idx == 10:
            break
        '''
    '''
    torch.cuda.synchronize()
    tb = time.perf_counter()
    print('time: {:.02e}s'.format((tb - ta)/(idx * 64)))
    '''
    test_loss = running_loss / len(dset_loader.dataset)
    test_acc = running_corrects.double() / len(dset_loader.dataset)

    if logger_name is not None:
        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format( \
                            'Test', test_loss, test_acc))

def main():
    model_folder = '../exp/cub/bcnnvd_cub3/checkpoints'
    model_path = os.path.join(model_folder, 'model_best.pth.tar')
    dataset = 'cub'
    crop_from_size = [448]
    input_size = [448]
    batch_size = 64
    model_names_list = ['vgg']
    pooling_method = 'outer_product' 
    embedding = 8192
    order = 2
    matrix_sqrt_iter = 0
    fc_bottleneck = False
    demo_agg = False
    # TODO: need the meta file including: crop_from_size, input_size,
    # model_names_list, pooling_method, fine_tune, pre_train, embedding, order 
    # matrix_sqrt_iter, fc_bottleneck, demo_agg
    
    data_transforms = [transforms.Compose([
            transforms.Resize(x[0]),
            transforms.CenterCrop(x[1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
            for x in zip(crop_from_size, input_size)]

    if dataset == 'cub':
        from CUBDataset import CUBDataset as Dataset
    elif dataset == 'cars':
        from CarsDataset import CarsDataset as Dataset
    elif dataset == 'aircrafts':
        from AircraftsDataset import AircraftsDataset as Dataset
    elif dataset == 'inat':
        from iNatDataset import iNatDataset as Dataset
    else:
        raise ValueError('Unknown dataset: %s' % task)

    dset_test = Dataset(dset_root[dataset], 'test', 
                    transform=data_transforms) 
    test_loader = torch.utils.data.DataLoader(dset_test,
                        batch_size=batch_size, shuffle=False,
                        num_workers=8, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_bcnn_model(model_names_list, len(dset_test.classes), 
                    pooling_method, False, True, embedding, order,
                    m_sqrt_iter=matrix_sqrt_iter, demo_agg=demo_agg,
                    fc_bottleneck=fc_bottleneck)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}')".format(model_path))

    test_model(model, criterion, test_loader, None)

if __name__ == '__main__':
    main()
