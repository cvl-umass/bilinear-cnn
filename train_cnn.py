import torch
import torch.nn as nn
import torch.optim as optim
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
from CNN import create_cnn_model
from test import test_model
from plot_curve import plot_log
import json

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
            # if name == 'module.fc.bias' or name == 'module.fc.weight':
            if 'module.fc' in name:
                fc_params_to_update.append(param)
            else:
                params_to_update.append(param)
            param.requires_grad = True

        # Observe that all parameters are being optimized
        if optimizer == 'sgd':
            '''
            optimizer_ft = optim.SGD([
                {'params': params_to_update},
                {'params': fc_params_to_update, 'weight_decay': 1e-5, 'lr': 1e-2}],
                lr=lr, momentum=0.9, weight_decay=wd)
            '''
            optimizer_ft = optim.SGD([
                {'params': params_to_update},
                {'params': fc_params_to_update}],
                lr=lr, momentum=0.9, weight_decay=wd)
        elif optimizer == 'adam':
            optimizer_ft = optim.Adam([
                {'params': params_to_update},
                {'params': fc_params_to_update, 'weight_decay': 1e-5, 'lr': 1e-2}],
                lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError('Unknown optimizer: %s' % optimizer)
    else:
        for name,param in model_ft.named_parameters():
            # if name == 'module.fc.bias' or name == 'module.fc.weight':
            if 'module.fc' in name:
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
        # maxItr=50000, logger_name='train_logger', checkpoint_folder='exp',
        epoch=45, logger_name='train_logger', checkpoint_folder='exp',
        start_itr=0, clip_grad=-1, scheduler=None):

    maxItr = epoch * len(dset_loader['train'].dataset) // \
                    dset_loader['train'].batch_size + 1

    val_every_number_examples = max(10000,
                    len(dset_loader['train'].dataset) // 5)
    val_frequency = val_every_number_examples // dset_loader['train'].batch_size 
    # val_frequency = 10000 // dset_loader['train'].batch_size 
    logger = logging.getLogger(logger_name)
    logger_filename = logger.handlers[1].stream.name

    device = next(model.parameters()).device
    since = time.time()

    running_loss = 0.0; running_num_data = 0 
    running_corrects = 0
    val_loss_history = []; best_acc = 0.0 
    val_acc = 0.0
    # best_model_wts = copy.deepcopy(model.state_dict())

    dset_iter = {x:iter(dset_loader[x]) for x in ['train', 'val']}
    bs = dset_loader['train'].batch_size
    update_frequency = batch_size_update // bs 
    model.train()
    last_epoch = 0 
    import pdb
    pdb.set_trace()
    for itr in range(start_itr, maxItr):
        # at the end of validation set model.train()
        if (itr + 1) % val_frequency == 0 or itr == maxItr - 1:
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

        inputs = inputs[0].to(device)
        # inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        with torch.set_grad_enabled(True):
            '''
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            ta = time.perf_counter()
            '''
            outputs = model(inputs)
            # outputs = model(*inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            '''
            torch.cuda.synchronize()
            tb = time.perf_counter()
            print('time: {:.02e}s'.format((tb - ta)/outputs.shape[0]))
            '''

            if (itr + 1) % update_frequency == 0:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        epoch = ((itr + 1) *  bs) // len(dset_loader['train'].dataset)

        '''
        running_num_data += inputs[0].size(0) 
        running_loss += loss.item() * inputs[0].size(0)
        '''
        running_num_data += inputs.size(0) 
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if (itr + 1) % val_frequency == 0 or itr == maxItr - 1:
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

                inputs = inputs[0].to(device)
                # inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    # outputs = model(*inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                # val_running_loss += loss.item() * inputs[0].size(0)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
            val_loss = val_running_loss / len(dset_loader['val'].dataset)
            val_acc = val_running_corrects.double() / len(dset_loader['val'].dataset)
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format('Validation',
            #                     val_loss, val_acc))
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format( \
                                'Validation', val_loss, val_acc))

            plot_log(logger_filename,
                    logger_filename.replace('history.txt', 'curve.png'), True)

            model.train()

        # update scheduler
        if scheduler is not None:
            if isinstance(scheduler, \
                    torch.optim.lr_scheduler.ReduceLROnPlateau):
                if (itr + 1) % val_frequency == 0: 
                    scheduler.step(val_acc)
            else:
                if epoch > last_epoch and scheduler is not None:
                    last_epoch = epoch
                    scheduler.step()
        # checkpoint
        if (itr + 1) & val_frequency == 0 or itr == maxItr - 1:
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

            checkpoint_dict = {
                'itr': itr + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_acc':  best_acc
            }
            if scheduler is not None:
                checkpoint_dict['scheduler'] = scheduler.state_dict()
            save_checkpoint(checkpoint_dict,
                    is_best, checkpoint_folder=checkpoint_folder)


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val accuracy: {:4f}'.format(best_acc))

    # load best model weights
    best_model_wts = torch.load(os.path.join(checkpoint_folder,
                                    'model_best.pth.tar'))
    model.load_state_dict(best_model_wts['state_dict'])
    # model.load_state_dict(best_model_wts)

    # return model, val_acc_history
    return model

def main(args):
    fine_tune = True 
    pre_train = True

    lr = args.lr
    input_size = args.input_size 
    # input_size = [448]
    # keep_aspect = True
    # model_names_list = ['vgg']
    # tensor_sketch = False
    # embedding = 8192

    model_names= args.model_names

    args.exp_dir = os.path.join(args.dataset, args.exp_dir)

    if args.dataset in ['cars', 'aircrafts']:
        keep_aspect = False
    else:
        keep_aspect = True

    if args.dataset in ['aircrafts']:
        crop_from_size = [(x * 256) // 224 for x in input_size]
    else:
        crop_from_size = input_size
        
    if 'inat' in args.dataset:
        split = {'train': 'train', 'val': 'val'}
    else:
        split = {'train': 'train_val', 'val': 'test'}

    if not keep_aspect:
        input_size = [(x, x) for x in input_size]
        crop_from_size = [(x, x) for x in crop_from_size]

    exp_root = '../exp'
    checkpoint_folder = os.path.join(exp_root, args.exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    args_dict = vars(args)
    with open(os.path.join(exp_root, args.exp_dir, 'args.txt'), 'a') as f:
        f.write(json.dumps(args_dict, sort_keys=True, indent=4))

    # make sure the dataset is ready
    if 'inat' in args.dataset:
        setup_dataset('inat')
    else:
        setup_dataset(args.dataset)

    # ==================  Craete data loader ==================================
    data_transforms = {
        'train': [transforms.Compose([
            transforms.Resize(x[0]),
            # transforms.CenterCrop(x[1]),
            transforms.RandomCrop(x[1]), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
            for x in zip(crop_from_size, input_size)],
        'val': [transforms.Compose([
            transforms.Resize(x[0]),
            transforms.CenterCrop(x[1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \
            for x in zip(crop_from_size, input_size)],
    }

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
                        transform=data_transforms[x]) for x in ['train', 'val']}
        dset_test = dataset(dset_root['inat'], 'test', subset, \
                        transform=data_transforms['val']) 
    else:
        dset = {x: dataset(dset_root[args.dataset], split[x], 
                        transform=data_transforms[x]) for x in ['train', 'val']}
        dset_test = dataset(dset_root[args.dataset], 'test', 
                        transform=data_transforms['val']) 

    dset_loader = {x: torch.utils.data.DataLoader(dset[x],
                batch_size=args.batch_size, shuffle=True, num_workers=4,
                drop_last=True) for x in ['train', 'val']} 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #======================= Initialize the model =========================
    
    # The argument embedding is used only when tensor_sketch is True
    # The argument order is used only when the model parameters are shared
    # between feature extractors
    model = create_cnn_model(model_names, len(dset['train'].classes),
                    input_size[0], fine_tune, pre_train)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    #====================== Initialize optimizer ==============================
    start_itr = 0

    if fine_tune:
        optim = initialize_optimizer(model, args.lr, optimizer=args.optimizer,
                                    wd=args.wd, finetune_model=fine_tune)

        if 'inat' not in args.dataset:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim,
                                lr_lambda=lambda epoch: 0.1 ** (epoch // 25))
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')
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
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (iteration{})"
                      .format(checkpoint_filename, checkpoint['itr']))


        # parallelize the model if using multiple gpus
        # if torch.cuda.device_count() > 1:
            
        # Train the miodel
        model = train_model(model, dset_loader, criterion, optim,
                batch_size_update=args.batch_size_update_model,
                # maxItr=args.iteration, logger_name=logger_name,
                epoch=args.epoch, logger_name=logger_name,
                checkpoint_folder=checkpoint_folder,
                start_itr=start_itr, scheduler=scheduler)

    if 'inat' not in args.dataset:
        # do test
        test_loader = torch.utils.data.DataLoader(dset_test,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=8, drop_last=False)
        print('evaluating test data')
        test_model(model, criterion, test_loader, logger_name)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_update_model', default=128, type=int,
            help='optimizer update the model after seeing batch_size number \
                    of inputs')
    parser.add_argument('--batch_size', default=32, type=int,
            help='size of mini-batch that can fit into gpus (sub bacth size')
    parser.add_argument('--epoch', default=45, type=int,
            help='number of epochs')
    parser.add_argument('--init_epoch', default=55, type=int,
            help='number of epochs for initializing fc layer')
    # parser.add_argument('--iteration', default=20000, type=int,
    #         help='number of iterations')
    parser.add_argument('--init_lr', default=1.0, type=float,
            help='learning rate')
    parser.add_argument('--lr', default=1e-2, type=float,
            help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float,
            help='weight decay')
    parser.add_argument('--init_wd', default=1e-8, type=float,
            help='weight decay for initializing fc layer')
    parser.add_argument('--optimizer', default='sgd', type=str,
            help='optimizer sgd|adam')
    parser.add_argument('--exp_dir', default='exp', type=str,
            help='foldername where to save the results for the experiment')
    parser.add_argument('--train_from_beginning', action='store_true',
            help='train the model from first epoch, i.e. ignore the checkpoint')
    # parser.add_argument('--train_split', default='train_val', type=str,
    #         help='split used to train augmentor')
    parser.add_argument('--dataset', default='cub', type=str,
            help='cub | cars | aircrafts')
    parser.add_argument('--input_size', nargs='+', default=[448], type=int,
            help='input size as a list of sizes')
    parser.add_argument('--model_names', default='vgg',
            type=str, help='input size as a list of sizes')
    parser.add_argument('--sketch', action='store_true',
            help='approximate tensor product in sketch space')
    parser.add_argument('--embedding_dim', type=int, default=8192,
            help='the dimension for the tnesor sketch approximation')
    parser.add_argument('--matrix_sqrt_iter', type=int, default=0,
            help='number of iteration for the Newtons Method approximating' + \
                    'matirx square rooti. Default=0 [no matrix square root]')
    parser.add_argument('--demo_agg', action='store_true',
            help='normalization with democratic aggregation')
    parser.add_argument('--fc_bottleneck', action='store_true',
            help='add bottelneck to the fc layers')
    args = parser.parse_args()

    main(args)

