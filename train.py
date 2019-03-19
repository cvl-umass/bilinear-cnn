import torch
import torch.nn as nn
import torch.optim as optim
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
from test import test_model
from plot_curve import plot_log

def initializeLogging(log_filename, logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.FileHandler(log_filename, mode='a'))

    return log

def save_checkpoint(state, is_best, checkpoint_folder='exp',
                filename='checkpoint.pth.tar'):
    filename = os.path.join(checkpoint_folder, filename)
    best_model_filename = os.path.join(checkpoint_folder, 'model_best.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)

# def initialize_optimizer(model_ft, lr, optimizer='sgd', finetune_model=True):
def initialize_optimizer(model_ft, lr, optimizer='sgd', wd=0, finetune_model=True):
    fc_params_to_update = []
    params_to_update = []
    if finetune_model:
        for name,param in model_ft.named_parameters():
            if name == 'fc.bias' or name == 'fc.weight':
                fc_params_to_update.append(param)
            else:
                params_to_update.append(param)
            param.requires_grad = True

        # Observe that all parameters are being optimized
        if optimizer == 'sgd':
            optimizer_ft = optim.SGD([
                {'params': params_to_update},
                {'params': fc_params_to_update, 'weight_decay': 0}],
                lr=lr, momentum=0.3, weight_decay=wd)
        elif optimizer == 'adam':
            optimizer_ft = optim.Adam([
                {'params': params_to_update},
                {'params': fc_params_to_update, 'weight_decay': 0}],
                lr=lr, momentum=0.3, weight_decay=wd)
        else:
            raise ValueError('Unknown optimizer: %s' % optimizer)
    else:
        for name,param in model_ft.named_parameters():
            if name == 'fc.bias' or name == 'fc.weight':
                param.requires_grad = True
                fc_params_to_update.append(param)
            else:
                param.requires_grad = False 

        # Observe that all parameters are being optimized
        if optimizer == 'sgd':
            optimizer_ft = optim.SGD(fc_params_to_update, lr=lr, momentum=0.9, 
                                weight_decay=wd)
        elif optimizer == 'adam':
            optimizer_ft = optim.Adam(fc_params_to_update, lr=lr, weight_decay=wd)
        else:
            raise ValueError('Unknown optimizer: %s' % optimizer)

    return optimizer_ft

def train_model(model, dset_loader, criterion,
        optimizer, batch_size_update=256,
        maxItr=50000, logger_name='train_logger', checkpoint_folder='exp',
        start_itr=0, clip_grad=-1):

    val_frequency = 10000 // dset_loader['train'].batch_size 
    logger = logging.getLogger(logger_name)
    logger_filename = logger.handlers[1].stream.name

    device = next(model.parameters()).device
    since = time.time()

    running_loss = 0.0; running_num_data = 0 
    running_corrects = 0
    val_loss_history = []; best_acc = 0.0 
    best_model_wts = copy.deepcopy(model.state_dict())

    dset_iter = {x:iter(dset_loader[x]) for x in ['train', 'val']}
    update_frequency = batch_size_update // dset_loader['train'].batch_size
    model.train()
    for itr in range(start_itr, maxItr):
        # at the end of validation set model.train()
        if (itr + 1) % val_frequency == 0:
            logger.info('Iteration {}/{}'.format(itr, maxItr - 1))
            logger.info('-' * 10)

        try:
            all_fields = next(dset_iter['train'])
            labels = all_fields[-2]
            inputs = all_fields[:-2]
            # inputs, labels, _ = next(dset_iter['train'])
        except StopIteration:
            dset_iter['train'] = iter(dset_loader['train'])
            all_fields = next(dset_iter['train'])
            labels = all_fields[-2]
            inputs = all_fields[:-2]
            # inputs, labels, _ = next(dset_iter['train'])

        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        with torch.set_grad_enabled(True):
            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            
            loss.backward()

            if (itr + 1) % update_frequency == 0:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        running_num_data += inputs[0].size(0) 
        running_loss += loss.item() * inputs[0].size(0)
        running_corrects += torch.sum(preds == labels.data)

        if (itr + 1) % val_frequency == 0:
            running_loss = running_loss / running_num_data
            running_acc = running_corrects.double() / running_num_data
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train',
            #                     running_loss, running_acc))
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format( \
                                'Train', running_loss, running_acc))
            running_loss = 0.0; running_num_data = 0; running_corrects = 0

            model.eval()
            val_running_loss = 0.0; val_running_corrects = 0

            # for inputs, labels, _ in dset_loader['val']:
            for all_fields in dset_loader['val']:
                labels = all_fields[-2]
                inputs = all_fields[:-2]
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(*inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                val_running_loss += loss.item() * inputs[0].size(0)
                val_running_corrects += torch.sum(preds == labels.data)
            val_loss = val_running_loss / len(dset_loader['val'].dataset)
            val_acc = val_running_corrects.double() / len(dset_loader['val'].dataset)
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format('Validation',
            #                     val_loss, val_acc))
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format( \
                                'Validation', val_loss, val_acc))

            plot_log(logger_filename,
                    logger_filename.replace('history.txt', 'curve.png'), False)

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            save_checkpoint({
                'itr': itr + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_acc':  best_acc,
            }, is_best, checkpoint_folder=checkpoint_folder)
            model.train()


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # return model, val_acc_history
    return model

def main(args):
    split = {'train':args.train_split, 'val':'test'}
    lr = args.lr
    input_size = [448]
    keep_aspect = True
    model_names_list = ['vgg']
    tensor_sketch = False 
    fine_tune = True 
    pre_train = True
    order = 2
    embedding = 8192

    if len(input_size) > 1:
        assert order == len(input_size)

    if not keep_aspect:
        input_size = [(x, x) for x in input_size]

    exp_root = '../exp'
    checkpoint_folder = os.path.join(exp_root, args.exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    init_checkpoint_folder = os.path.join(exp_root, args.exp_dir, 'init_checkpoints')
    if not os.path.isdir(init_checkpoint_folder):
        os.makedirs(init_checkpoint_folder)

    args_dict = vars(args)
    import json
    with open(os.path.join(exp_root, args.exp_dir, 'args.txt'), 'a') as f:
        f.write(json.dumps(args_dict, sort_keys=True, indent=4))



    # ==================  Craete data loader ==================================
    if keep_aspect:
        data_transforms = {
            'train': [transforms.Compose([
                transforms.Resize(x),
                transforms.CenterCrop(x),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
                for x in input_size],
            'val': [transforms.Compose([
                transforms.Resize(x),
                transforms.CenterCrop(x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
                for x in input_size]
        }
    else:
        data_transforms = {
            'train': [transforms.Compose([
                transforms.Resize(x),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
                for x in input_size],
            'val': [transforms.Compose([
                transforms.Resize(x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
                for x in input_size]
        }

    # TODO: update the data loaders to take a list of transformantions
    # prepare the training tasks and data loaders
    if args.dataset == 'cub':
        from CUBDataset import CUBDataset
        dset = {x: CUBDataset(dset_root['cub'], split[x], 
                            transform=data_transforms[x]) \
                for x in ['train', 'val']}
        dset_test = CUBDataset(dset_root['cub'], 'test', 
                            transform=data_transforms['val']) 
    elif args.dataset == 'cars':
        from CarsDataset import CarsDataset
        dset = {x: CarsDataset(dset_root['cars'], split[x],
                            transform=data_transforms[x]) \
                for x in ['train', 'val']}
        dset_test = CarsDataset(dset_root['cars'], 'test',
                            transform=data_transforms['val'])
    elif args.dataset == 'aircrafts':
        from AircraftsDataset import AircraftsDataset
        dset = {x: AircraftsDataset(dset_root['aircrafts'], split[x],
                            transform=data_transforms[x]) \
                for x in ['train', 'val']}
        dset_test = AircraftsDataset(dset_root['aircrafts'], 'test',
                            transform=data_transforms['val'])
    else:
        raise ValueError('Unknown dataset: %s' % task)
    '''
    elif task[:len('inat_')] == 'inat_':
        from iNatDataset import iNatDataset
        subtask = task[len('inat_'):]
        subtask = subtask[0].upper() + subtask[1:]
        dset_list.append(iNatDataset(dset_root['inat'], split, subtask, 
            load_val_boxes=True, transform=data_transforms['train']))
    '''

    dset_loader = {x: torch.utils.data.DataLoader(dset[x],
                batch_size=args.batch_size, shuffle=True, num_workers=4,
                drop_last=True) for x in ['train', 'val']} 


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #======================= Initialize the model =========================
    
    # The argument embedding is used only when tensor_sketch is True
    # The argument order is used only when the model parameters are shared
    # between feature extractors
    model = create_bcnn_model(model_names_list, len(dset['train'].classes), 
                    tensor_sketch, fine_tune, pre_train, embedding, order)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    #====================== Initialize optimizer ==============================
    init_model_checkpoint = os.path.join(init_checkpoint_folder,
                                        'checkpoint.pth.tar')
    start_itr = 0
    if not args.train_from_beginning:
        logger_name = 'train_init_logger'
        logger = initializeLogging(os.path.join(exp_root, args.exp_dir, 
                'train_init_history.txt'), logger_name)

        if os.path.isfile(init_model_checkpoint):
            print("=> loading checkpoint '{}'".format(init_model_checkpoint))
            checkpoint = torch.load(init_model_checkpoint)
            start_itr = checkpoint['itr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint for the fc initialization")

    optim_fc = initialize_optimizer(model, 1.0, optimizer='sgd', wd=1e-8,
                                finetune_model=False)
    model = model.to(device)
    model = train_model(model, dset_loader, criterion, optim_fc,
            batch_size_update=256,
            maxItr=40000, logger_name=logger_name,
            start_itr=start_itr,
            checkpoint_folder=init_checkpoint_folder)

    if fine_tune:
        model = torch.nn.DataParallel(model)
        optim = initialize_optimizer(model, args.lr, optimizer=args.optimizer,
                                    wd=args.wd, finetune_model=fine_tune)

        logger_name = 'train_logger'
        logger = initializeLogging(os.path.join(exp_root, args.exp_dir, 
                'train_history.txt'), logger_name)

        start_itr = 0
        # load from checkpoint if exist
        if not args.train_from_beginning:
            checkpoint_filename = os.path.join(checkpoint_folder,
                        'checkpoint.pth.tar')
            if os.path.isfile(checkpoint_filename):
                print("=> loading checkpoint '{}'".format(checkpoint_filename))
                checkpoint = torch.load(checkpoint_filename)
                start_itr = checkpoint['itr']
                model.load_state_dict(checkpoint['state_dict'])
                optim.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (iteration{})"
                      .format(checkpoint_filename, checkpoint['itr']))

        # parallelize the model if using multiple gpus
        # if torch.cuda.device_count() > 1:
            
        # Train the miodel
        model = train_model(model, dset_loader, criterion, optim,
                batch_size_update=args.batch_size_update_model,
                maxItr=args.iteration, logger_name=logger_name,
                checkpoint_folder=checkpoint_folder,
                start_itr=start_itr)
    # do test
    test_loader = torch.utils.data.DataLoader(dset_test,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=8, drop_last=False)
    test_model(model, criterion, test_loader, logger_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_update_model', default=128, type=int,
            help='optimizer update the model after seeing batch_size number \
                    of inputs')
    parser.add_argument('--batch_size', default=32, type=int,
            help='size of mini-batch that can fit into gpus (sub bacth size')
    parser.add_argument('--iteration', default=20000, type=int,
            help='number of iterations')
    parser.add_argument('--lr', default=1e-2, type=float,
            help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float,
            help='weight decay')
    parser.add_argument('--optimizer', default='sgd', type=str,
            help='optimizer sgd|adam')
    parser.add_argument('--exp_dir', default='exp', type=str,
            help='foldername where to save the results for the experiment')
    parser.add_argument('--train_from_beginning', action='store_true',
            help='train the model from first epoch, i.e. ignore the checkpoint')
    parser.add_argument('--train_split', default='train_val', type=str,
            help='split used to train augmentor')
    parser.add_argument('--dataset', default='cub', type=str,
            help='cub | cars | aircrafts')
    args = parser.parse_args()

    main(args)

