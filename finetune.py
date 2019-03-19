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
from tensorboardX import SummaryWriter
from stn import STNet

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

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_optimizer(model_ft, feature_extract=False, stn=False):
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(params_to_update, lr=1e-4, weight_decay=0, betas=(0.9, 0.999))
    if stn is False:
        optimizer_ft = optim.Adam(params_to_update, lr=1e-4, weight_decay=0, betas=(0.9, 0.999))
    else:
        params_to_update = []
        # params_to_update_name = []
        for name,param in model_ft.model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # params_to_update_name.append(name)
        params_to_update_stn = []
        # params_to_update_stn_name = []
        for name,param in model_ft.fc_loc.named_parameters():
            if param.requires_grad == True:
                params_to_update_stn.append(param)
                # params_to_update_stn_name.append(name)
        for name,param in model_ft.localization.named_parameters():
            if param.requires_grad == True:
                params_to_update_stn.append(param)
                # params_to_update_stn_name.append(name)

        optimizer_ft = optim.Adam([ {'params':params_to_update},
                                    {'params':params_to_update_stn, 'lr':1e-8, 'weight_decay':1e-5}],
                                    lr=1e-4, weight_decay=0, betas=(0.9, 0.999))
    return optimizer_ft

def initialize_model(model_name, num_classes, feature_extract=False,
                    use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        # print("Invalid model name, exiting...")
        logger.debug("Invalid mode name")
        exit()

    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs=35,
    is_inception=False, logger_name='train_logger', checkpoint_folder='exp',
    start_epoch=0, writer=None):

    logger = logging.getLogger(logger_name)

    device = next(model.parameters()).device
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.info('-' * 10)
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logger.info('{} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc*100))

            writer.add_scalar(phase+'/loss', epoch_loss, epoch+1)
            writer.add_scalar(phase+'/acc', epoch_acc*100, epoch+1)

            # deep copy the model
            is_best = epoch_acc > best_acc
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_folder=checkpoint_folder)

        if epoch > 0 and (epoch+1) % 15 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.5


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    writer.close()
    return model, val_acc_history

def main(args):

    log_dir = args.exp_dir+'/log'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    batch_size = 32
    maxIter = 10000
    split = 'val'
    input_size = 224

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    if not os.path.isdir(os.path.join(args.exp_dir, args.task)):
        os.makedirs(os.path.join(args.exp_dir, args.task))
    checkpoint_folder = os.path.join(args.exp_dir, args.task, 'checkpoints')
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    logger_name = 'train_logger'
    logger = initializeLogging(os.path.join(args.exp_dir, args.task, 
            'train_history.txt'), logger_name)

    # ==================  Craete data loader ==================================
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size), 
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    if args.task == 'cub':
        from CUBDataset import CUBDataset
        image_datasets = {split: CUBDataset(dset_root['cub'], split,
            create_val=True, transform=data_transforms[split]) \
            for split in ['train', 'val']}
    elif args.task == 'cars':
        from CarsDataset import CarsDataset
        image_datasets = {split: CarsDataset(dset_root['cars'], split,
            create_val=True, transform=data_transforms[split]) \
            for split in ['train', 'val']}
    elif args.task == 'aircrafts':
        from AircraftsDataset import AircraftsDataset
        image_datasets = {split: AircraftsDataset(dset_root['aircrafts'], split,
            transform=data_transforms[split]) \
            for split in ['train', 'val']}
    elif args.task[:len('inat_')] == 'inat_':
        from iNatDataset import iNatDataset
        task = args.task
        subtask = task[len('inat_'):]
        subtask = subtask[0].upper() + subtask[1:]
        image_datasets = {split: iNatDataset(dset_root['inat'], split, subtask,
            transform=data_transforms[split]) \
            for split in ['train', 'val']}
    else:
        raise ValueError('Unknown dataset: %s' % task)


    num_classes = image_datasets['train'].get_num_classes()

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                batch_size=args.batch_size, shuffle=True, num_workers=4) \
                for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #======================= Initialize the model==============================
    model_ft, input_size = initialize_model(args.model, num_classes, 
                    feature_extract=False, use_pretrained=True)
    if args.stn:
        model_ft = STNet(model_ft)
    model_ft = model_ft.to(device)

    #====================== Initialize optimizer ==============================
    optim = initialize_optimizer(model_ft, feature_extract=False, stn=args.stn)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    # load from checkpoint if exist
    if not args.train_from_beginning:
        checkpoint_filename = os.path.join(checkpoint_folder,
                    'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_filename):
            print("=> loading checkpoint '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename)
            start_epoch = checkpoint['epoch']
            best_acc= checkpoint['best_acc']
            model_ft.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_filename, checkpoint['epoch']))

    # parallelize the model if using multiple gpus
    if torch.cuda.device_count() > 1:
        model_ft = torch.nn.DataParallel(model_ft)
        
    # Train the miodel
    model_ft = train_model(model_ft, dataloaders_dict, criterion, optim,
            num_epochs=args.num_epochs, is_inception=(args.model=="inception"),
            logger_name=logger_name, checkpoint_folder=checkpoint_folder,
            start_epoch=start_epoch, writer=writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cub', type=str, 
            help='the name of the task|dataset')
    parser.add_argument('--model', default='resnet50', type=str,
            help='resnet|densenet')
    parser.add_argument('--batch_size', default=32, type=int,
            help='size of mini-batch')
    parser.add_argument('--num_epochs', default=35, type=int,
            help='number of epochs')
    parser.add_argument('--exp_dir', default='exp', type=str,
            help='path to the chekcpoint folder for the experiment')
    parser.add_argument('--train_from_beginning', action='store_true',
            help='train the model from first epoch, i.e. ignore the checkpoint')
    parser.add_argument('--stn', dest='stn', action='store_true',
            help='use STN')
    args = parser.parse_args()
    main(args)



