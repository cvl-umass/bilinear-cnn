import torch
# import cv2
import matplotlib.pyplot as plt
import numpy as np
# import lmdb
import os
import json
import math
import PIL
from config import dset_root
from shutil import copyfile


'''
base_file_list = ['./filelists/metaiNat/base.json',
                    './filelists/metaiNat/val.json',
                    './filelists/metaiNat/novel.json',
                    './filelists/metaiNat/novel_train.json',
                    './filelists/metaiNat/novel_test.json']
'''

all_json = ['train2018.json', 'val2018.json', 'test2018.json']

to_size = 448 
output_root = dset_root['inat'].replace('inat_2018', 'inat_2018_%d'%to_size)
                    
for json_file in all_json:
# for base_file in base_file_list:

    base_file = os.path.join(dset_root['inat'], json_file)
    with open(base_file, 'r') as f:
        meta = json.load(f)

    folder_name = set([os.path.dirname(x['file_name']) for x in meta['images']])

    for x in folder_name:
        fpath = os.path.join(output_root, x)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)

    '''
    lmdb_file, _ = os.path.splitext(base_file)
    base_name = os.path.basename(lmdb_file)
    dataset = SimpleDataset(base_file, None)
    '''

    num_img = len(meta['images'])

    for idx, x in enumerate(meta['images']):
        im = PIL.Image.open(os.path.join(dset_root['inat'], x['file_name']))
        ratio = to_size / min(im.size)
        resize_to = tuple([math.ceil(y*ratio) for y in im.size])
        resizeImg = im.resize(resize_to, resample=PIL.Image.BILINEAR)

        out_name = os.path.join(output_root, x['file_name'])
        resizeImg.save(out_name)

        if (idx + 1) % 1000 == 0:
            print('%s: %d / %d'%(json_file.split('.')[0], idx+1, num_img))

    copyfile(base_file, os.path.join(output_root, json_file))












    
