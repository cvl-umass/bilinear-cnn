## Overview
This repository contains the code re-implementing the following papers in Pytorch. 
The links to their project webpages are provided and their original imeplementations in Matlab with MatConVnet can be found in the project webpages.

1. [Bilinear Convolutional Neural Networks for Fine-grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/), Tsung-Yu Lin, Aruni RoyChowdhury and Subhransu Maji, PAMI 2017
2. [Visualizing and Understanding Deep Texture Representations](http://vis-www.cs.umass.edu/bcnn/), Tsung-Yu Lin, and Subhransu Maji, CVPR 2016
3. [Improved Bilinear Pooling with CNNs](http://vis-www.cs.umass.edu/bcnn/), Tsung-Yu Lin, and Subhransu Maji, BMVC 2017
4. [Second-order Democratic Aggregation](http://vis-www.cs.umass.edu/o2dp/), Tsung-Yu Lin, Subhransu Maji and Piotr Koniusz, ECCV 2018

The series of works investigate the models using second-order pooling of convolutional features and study the techniques to improve the representation power.
More details can be found in my PhD [thesis](http://vis-www.cs.umass.edu/papers/tsungyu_thesis.pdf).

In this repository, we provided the code for:
1. training BCNN models
2. training Improved BCNN models (Improving BCNN with matrix square root normalization)
3. training the CNN models with second-order democratic aggregation
4. inverting fine-grained categories with BCNN representations

## Datasets
Download the following datasets and point the correpsonding entries in `config.py` to the location where you download the data.
* Birds: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
* Aircrafts: [FGVC aircraft dataset](http://www.robots.ox.ac.uk/~vgg/data/oid/)
* Cars: [Stanford cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

