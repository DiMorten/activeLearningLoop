import os, errno
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


from glob import glob
from PIL import Image
from torchvision import transforms, utils

from FOD.Loss import ScaleAndShiftInvariantLoss
from FOD.Custom_augmentation import ToMask

import pdb, sys
from icecream import ic
import cv2

def getFilesWithoutBlankReference(dataset_name, files):
    path = 'C:/Users/jchamorro/Documents/petrobras/Darwin/'
    files_filtered = []
    print("len(files)", len(files))
    for file in files:
        # print("file", path + dataset_name + '/depths/' + file)
        # pdb.set_trace()
        im = cv2.imread(path + dataset_name + '/depths/' + file)
        if np.any(im != 0):
            files_filtered.append(file)
    print("len(files_filtered)", len(files_filtered))
    # pdb.set_trace()
    return files_filtered

        
def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, split, dataset_name, path_images, path_depths, path_segmentation):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['General']['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])]# [:100]
        # selected_files = getFilesWithoutBlankReference(dataset_name, selected_files)
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])]
        # selected_files = getFilesWithoutBlankReference(dataset_name, selected_files)
    else:
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):]# [:100]

    ic(os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images']))
    print('Train list', list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])])
    print('Val list', list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])])
    print('Test list', list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):])

    # exit(0)

    path_images = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'], im[:-4]+config['Dataset']['extensions']['ext_images']) for im in selected_files]
    path_depths = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_depths'], im[:-4]+config['Dataset']['extensions']['ext_depths']) for im in selected_files]
    path_segmentation = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'], im[:-4]+config['Dataset']['extensions']['ext_segmentations']) for im in selected_files]
    return path_images, path_depths, path_segmentation

def get_transforms(config):
    im_size = config['Dataset']['transforms']['resize']
    transform_image = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.Grayscale(num_output_channels=1) ,
        transforms.ToTensor()
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
        ToMask(config['Dataset']['classes']),
    ])
    return transform_image, transform_depth, transform_seg

def get_losses(config):
    def NoneFunction(a, b):
        return 0
    loss_depth = NoneFunction
    loss_segmentation = NoneFunction
    type = config['General']['type']
    if type == "full" or type=="depth":
        if config['General']['loss_depth'] == 'mse':
            loss_depth = nn.MSELoss()
        elif config['General']['loss_depth'] == 'ssi':
            loss_depth = ScaleAndShiftInvariantLoss()
    if type == "full" or type=="segmentation":
        if config['General']['loss_segmentation'] == 'ce':
            # weights = [0.6, 3.7]
            # weights = [1, 2]
            
            # class_weights = torch.FloatTensor(weights).cuda()
            loss_segmentation = nn.CrossEntropyLoss()
    return loss_depth, loss_segmentation

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# def get_optimizer(config, net):
#     if config['General']['optim'] == 'adam':
#         optimizer = optim.Adam(net.parameters(), lr=config['General']['lr'])
#     elif config['General']['optim'] == 'sgd':
#         optimizer = optim.SGD(net.parameters(), lr=config['General']['lr'], momentum=config['General']['momentum'])
#     return optimizer
'''
def get_optimizer(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'transformer_encoders'])
    params_backbone = net.transformer_encoders.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['General']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['General']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['General']['lr_scratch'])
    elif config['General']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['General']['lr_backbone'], momentum=config['General']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['General']['lr_scratch'], momentum=config['General']['momentum'])
    return optimizer_backbone, optimizer_scratch
'''
def get_optimizer(config, net):

    if config['General']['optim'] == 'adam':
    #     optimizer_backbone = optim.Adam(params_backbone, lr=config['General']['lr_backbone'])
        optimizer_scratch = optim.Adam(net.parameters(), lr=config['General']['lr_scratch'])
    elif config['General']['optim'] == 'sgd':
    #     optimizer_backbone = optim.SGD(params_backbone, lr=config['General']['lr_backbone'], momentum=config['General']['momentum'])
        optimizer_scratch = optim.SGD(net.parameters(), lr=config['General']['lr_scratch'], momentum=config['General']['momentum'])
    return None, optimizer_scratch


def get_schedulers(optimizers):
    optimizers = [optimizer for optimizer in optimizers if optimizer != None]
    return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]

def filterSamplesByIdxs(dataset, idxs):
    
    dataset.paths_images = np.array(dataset.paths_images)[idxs]
    dataset.paths_depths = np.array(dataset.paths_depths)[idxs]
    dataset.paths_segmentations = np.array(dataset.paths_segmentations)[idxs]
    return dataset

def filterSamplesByRandomIdxs(dataset, n):
    idxs = np.arange(len(dataset.paths_images))
    np.random.shuffle(idxs)
    idxs = idxs[:n]
    dataset.paths_images = np.array(dataset.paths_images)[idxs]
    dataset.paths_depths = np.array(dataset.paths_depths)[idxs]
    dataset.paths_segmentations = np.array(dataset.paths_segmentations)[idxs]
    return dataset
