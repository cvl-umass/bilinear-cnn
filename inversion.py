from config import dset_root, setup_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import argparse
import sys
from BCNN import create_multi_heads_bcnn 
import json
import logging
import copy
import shutil
import scipy.misc

def initializeLogging(log_filename, logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.FileHandler(log_filename, mode='a'))

    return log

def train_model(model, dset_loader, criterion,
        optimizer, batch_size_update=256,
        # maxItr=50000, logger_name='train_logger', checkpoint_folder='exp',
        epoch=45, logger_name='train_logger', checkpoint_folder='exp',
        start_itr=0, clip_grad=-1, scheduler=None, fine_tune=True):

    maxItr = epoch * len(dset_loader['train'].dataset) // \
                    dset_loader['train'].batch_size + 1

    val_every_number_examples = max(10000,
                    len(dset_loader['train'].dataset) // 5)
    val_frequency = val_every_number_examples // dset_loader['train'].batch_size
    checkpoint_frequency = 5 * len(dset_loader['train'].dataset) / \
                                dset_loader['train'].batch_size
    last_checkpoint = start_itr  - 1
    logger = logging.getLogger(logger_name)

    device = next(model.parameters()).device

    running_num_data = 0
    # Train the fc classifier for the features from 4 layers
    # {relu2_2, relu3_3, relur4_3, relu5_3}
    running_loss = [0.0] * 4
    running_corrects = [0] * 4

    best_acc = [0.0] * 4

    dset_iter = {x:iter(dset_loader[x]) for x in ['train', 'val']}
    bs = dset_loader['train'].batch_size
    update_frequency = batch_size_update // bs

    model.module.fc_list.train()

    last_epoch = 0
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

        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        with torch.set_grad_enabled(True):
            outputs = model(*inputs)
            loss_list = [criterion(output, labels) for output in outputs]
            loss = torch.sum(torch.stack(loss_list))

            preds = [] 
            for output in outputs:
                _, pred = torch.max(output, 1)
                preds.append(pred)

            loss.backward()

            if (itr + 1) % update_frequency == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch = ((itr + 1) *  bs) // len(dset_loader['train'].dataset)

        running_num_data += inputs[0].size(0)
        for idx, loss_ in enumerate(loss_list):
            running_loss[idx] += loss_.item() * inputs[0].size(0)
            running_corrects[idx] += torch.sum(preds[idx] == labels.data)

        if (itr + 1) % val_frequency == 0 or itr == maxItr - 1:
            running_loss = [
                    r_loss / running_num_data for r_loss in running_loss
            ]
            running_acc = [
                    r_corrects.double() / running_num_data
                    for r_corrects in running_corrects
            ]
            logger.info(
                '{} Loss: {:.4f} {:.4f} {:.4f} {:.4f} Acc: {:.4f} {:.4f} {:.4f} {:.4f}'.format( \
                'Train - relu2_2, relu3_3, relu4_3, relu5_3',
                *running_loss, *running_acc)
            )
            running_num_data = 0
            running_loss = [0.0] * 4
            running_corrects = [0] * 4

            model.eval()

            val_running_loss = [0.0] * 4
            val_running_corrects = [0] * 4

            for all_fields in dset_loader['val']:
                labels = all_fields[-2]
                inputs = all_fields[:-2]
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(*inputs)
                    loss_list = [criterion(output, labels) for output in outputs]
                    loss = torch.sum(torch.stack(loss_list))
            
                    preds = [] 
                    for output in outputs:
                        _, pred = torch.max(output, 1)
                        preds.append(pred)

                for idx, loss_ in enumerate(loss_list):
                    val_running_loss[idx] += loss_.item() * inputs[0].size(0)
                    val_running_corrects[idx] += torch.sum(preds[idx] == labels.data)

            val_loss = [
                    r_loss / len(dset_loader['val'].dataset)
                    for r_loss in val_running_loss
            ]
            val_acc = [
                    r_corrects.double() / len(dset_loader['val'].dataset)
                    for r_corrects in val_running_corrects
            ]
            logger.info(
                '{} Loss: {:.4f} {:.4f} {:.4f} {:.4f} Acc: {:.4f} {:.4f} {:.4f} {:.4f}'.format( \
                'Validation - relu2_2, relu3_3, relu4_3, relu5_3',
                *val_loss, *val_acc)
            )

            model.module.fc_list.train()

        # checkpoint
        if (itr + 1) % val_frequency == 0 or itr == maxItr - 1:
            do_checkpoint = (itr - last_checkpoint) >= checkpoint_frequency
            if do_checkpoint or itr == maxItr - 1:
                last_checkpoint = itr
                checkpoint_dict = {
                    'itr': itr + 1,
                    'state_dict': model.module.fc_list.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_acc':  best_acc
                }
                save_checkpoint(
                        checkpoint_dict,
                        checkpoint_folder=checkpoint_folder
                )
                
            best_model_path = os.path.join(checkpoint_folder, 'model_best.pth.tar')


            update_best_model = False
            for v_idx, val_acc_ in enumerate(val_acc):
                is_best = val_acc_ > best_acc[v_idx]
                if is_best:
                    if os.path.isfile(best_model_path):
                        best_fc = torch.load(best_model_path)
                        best_fc = best_fc['state_dict']
                    else:
                        best_fc = copy.deepcopy(
                                model.module.fc_list.state_dict()
                        )
                    update_best_model = True
                    break

            for v_idx, val_acc_ in enumerate(val_acc):
                is_best = val_acc_ > best_acc[v_idx]
                param_names = ['%d.weight'%v_idx, '%d.bias'%v_idx]
                if is_best:
                    best_acc[v_idx] = val_acc_
                    for name in param_names:
                        best_fc[name] = model.module.fc_list.state_dict()[name]

            if update_best_model:
                torch.save({'state_dict': best_fc}, best_model_path)

    logger.info('Best val accuracy: {:4f} {:4f} {:4f} {:4f}'.format(*best_acc))

    # load best model weights
    best_model_wts = torch.load(os.path.join(checkpoint_folder, 'model_best.pth.tar'))
    model.module.fc_list.load_state_dict(best_model_wts['state_dict'])

    return model

def save_checkpoint(
        state,
        checkpoint_folder='exp',
        filename='checkpoint.pth.tar'
):
    filename = os.path.join(checkpoint_folder, filename)
    torch.save(state, filename)


def initialize_optimizer(model_ft, lr, wd=0):

    fc_params_to_update = []
    fc_params_group_2 = []
    fc_params_group_3 = []
    for name, param in model_ft.named_parameters():
        # if name == 'module.fc.bias' or name == 'module.fc.weight':
        if 'module.fc_list' in name:
            param.requires_grad = True
            if '0' in name:
                fc_params_group_3.append(param)
            elif '1' in name:
                fc_params_group_2.append(param)
            else:
                fc_params_to_update.append(param)
        else:
            param.requires_grad = False

    '''
    optimizer_ft = optim.SGD(fc_params_to_update, lr=lr, momentum=0.9,
                        weight_decay=wd)
    '''
    optimizer_ft = optim.SGD([
        {'params': fc_params_to_update},
        {'params': fc_params_group_2, 'lr': lr * 1},
        {'params': fc_params_group_3, 'lr': lr * 1}],
        lr=lr, momentum=0.9, weight_decay=wd)

    return optimizer_ft


def inverting_categories(
        classes,
        model,
        criterion,
        input_size,
        tv_beta = 2,
        num_steps=200,
        logger_name='inv_logger',
):
    logger = logging.getLogger(logger_name)
    device = next(model.parameters()).device
    output_imgs = []
    for i in range(len(classes)):
        target_label = torch.tensor([i], dtype=torch.int64, device=device)
        logger.info('=' * 80 + '\nClass {}:'.format(classes[i]))
        img = torch.randn(
                [1, 3, *input_size],
                dtype=torch.float32,
                device=device,
                requires_grad=True
        )
        optimizer = optim.LBFGS([img])

        itr = 0
        while itr < num_steps:

            cache_loss = [0.0]
            def closure():
                optimizer.zero_grad()
                preds_softmax = model(img)

                inv_loss = [] 
                for output in preds_softmax:
                    loss_ = criterion(output, target_label) 
                    inv_loss.append(loss_)
                loss = torch.sum(torch.stack(inv_loss))

                d1 = img[:,:,1:,:] - img[:,:,:-1,:]
                d2 = img[:,:,:,1:] - img[:,:,:,:-1]
                tv = torch.sum(
                    (
                        torch.sqrt(
                            d1.view(-1) ** 2 +
                            d2.view(-1) ** 2
                        ) **
                        tv_beta
                    )
                )

                loss += 1e-9 * tv

                loss.backward()

                # logger.info('Loss: {:.4f}'.format(loss.item()))
                # current_loss[0] = loss.item()
                cache_loss[0] = loss.item()
                return loss

            # optimizer.step(lambda : closure(cache_loss))
            optimizer.step(closure)
            logger.info('Step {} Loss: {}'.format(itr, cache_loss[0]))
            itr += 1

        output_imgs.append(torch.squeeze(img))

    return output_imgs

def save_outputs(output_imgs, classes, output_folder):
    device = output_imgs[0].device
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    img_var = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    for img, c_name in zip(output_imgs, classes):
        img = img * img_var + img_mean
        img.data.clamp_(0, 1)
        img = img.permute(2, 1, 0).cpu().detach().numpy()
        x_range = np.percentile(img, [1, 99])
        img = np.clip(img, x_range[0], x_range[1])

        img = (img - x_range[0]) / (x_range[1] - x_range[0])

        output_file_name = os.path.join(output_folder, c_name + '.png') 
        scipy.misc.imsave(output_file_name, img)

def main(args):
    lr = args.lr
    input_size = args.input_size

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

    if len(input_size) > 1:
        assert order == len(input_size)

    if not keep_aspect:
        input_size = [(x, x) for x in input_size]
        crop_from_size = [(x, x) for x in crop_from_size]

    exp_root = '../exp_inversion'
    checkpoint_folder = os.path.join(exp_root, args.exp_dir, 'checkpoints')

    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # log the setup for the experiments
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
    else:
        dset = {x: dataset(dset_root[args.dataset], split[x], \
                        transform=data_transforms[x]) for x in ['train', 'val']}

    dset_loader = {x: torch.utils.data.DataLoader(dset[x],
                batch_size=32, shuffle=True, num_workers=8,
                drop_last=drop_last) \
                for x, drop_last in zip(['train', 'val'], [True, False])}


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #======================= Initialize the model =========================
    model = create_multi_heads_bcnn(len(dset['train'].classes)) 
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    #====================== Initialize optimizer ==============================
    model_checkpoint = os.path.join(checkpoint_folder, 'checkpoint.pth.tar')
    start_itr = 0
    optim_fc = initialize_optimizer(
            model,
            args.lr,
            wd=args.wd,
    )

    logger_name = 'train_logger'
    logger = initializeLogging(
                os.path.join(exp_root, args.exp_dir, 'train_fc_history.txt'),
                logger_name
    )

    model_train_fc = False
    fc_model_path = os.path.join(exp_root, args.exp_dir, 'fc_params.pth.tar')
    if not args.train_from_beginning:
        if os.path.isfile(fc_model_path):
            # load the fc parameters if they are already trained
            print("=> loading fc parameters'{}'".format(fc_model_path))
            checkpoint = torch.load(fc_model_path)
            model.module.fc_list.load_state_dict(checkpoint['state_dict'])
            print("=> loaded fc initialization parameters")
        else:
            if os.path.isfile(model_checkpoint):
                # load the checkpoint if it exists
                print("=> loading checkpoint '{}'".format(model_checkpoint))
                checkpoint = torch.load(model_checkpoint)
                start_itr = checkpoint['itr']
                model.module.fc_list.load_state_dict(checkpoint['state_dict'])
                optim_fc.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint for the fc initialization")

            # resume training
            model_train_fc = True
    else:
        # Training everything from the beginning
        model_train_fc = True
        start_itr = 0

    if model_train_fc:
        # do the training
        model.eval()

        model = train_model(model, dset_loader, criterion, optim_fc,
                batch_size_update=256,
                epoch=args.epoch, logger_name=logger_name, start_itr=start_itr,
                checkpoint_folder=checkpoint_folder, fine_tune=False)
        shutil.copyfile(
                os.path.join(checkpoint_folder, 'model_best.pth.tar'),
                fc_model_path)
    
    logger_inv = initializeLogging(
                os.path.join(exp_root, args.exp_dir, 'inv_history.txt'),
                'inv_logger' 
    )

    output_images = inverting_categories(
            dset['train'].classes,
            model,
            criterion,
            [224, 224],
            logger_name='inv_logger',
    )

    inv_folder = os.path.join(exp_root, args.exp_dir, 'inv_outputs')
    if not os.path.isdir(inv_folder):
        os.makedirs(inv_folder)
    save_outputs(output_images, dset['train'].classes, inv_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=45, type=int,
            help='number of epochs')
    parser.add_argument('--lr', default=1, type=float,
            help='learning rate')
    parser.add_argument('--wd', default=1e-8, type=float,
            help='weight decay')
    parser.add_argument('--exp_dir', default='inv', type=str,
            help='foldername where to save the results for the experiment')
    parser.add_argument('--train_from_beginning', action='store_true',
            help='train the model from first epoch, i.e. ignore the checkpoint')
    parser.add_argument('--dataset', default='cub', type=str,
            help='cub | cars | aircrafts')
    parser.add_argument('--input_size', nargs='+', default=[448], type=int,
            help='input size as a list of sizes')
    args = parser.parse_args()

    main(args)

