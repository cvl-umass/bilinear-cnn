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

def test_model(model, criterion,  dset_loader, logger_name):

    logger = logging.getLogger(logger_name)
    device = next(model.parameters()).device
    model.eval()

    running_loss = 0.0; running_corrects = 0
    for all_fields in dset_loader:
        labels = all_fields[-2]
        inputs = all_fields[:-2]
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs[0].size(0)
        running_corrects += torch.sum(preds == labels.data)
    test_loss = running_loss / len(dset_loader.dataset)
    test_acc = running_corrects.double() / len(dset_loader.dataset)

    logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format( \
                        'Test', test_loss, test_acc))
