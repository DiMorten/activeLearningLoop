import os, errno
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from torchvision import transforms, utils

import pdb, sys
from icecream import ic
import cv2
import pandas as pd
import pathlib
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

def get_splitted_dataset(config, split, input_folder_path, path_images, path_segmentation = None):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['split_train'])]# [:100]
        # selected_files = getFilesWithoutBlankReference(dataset_name, selected_files)
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['split_train']):int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val'])]
        # selected_files = getFilesWithoutBlankReference(dataset_name, selected_files)
    else:
        selected_files = list_files[int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val']):]# [:100]

    # print("os.path.join(input_folder_path, config['Dataset']['paths']['path_images'])", 
    #     os.path.join(input_folder_path, config['Dataset']['paths']['path_images']))
    print('Train list', list_files[:int(len(list_files)*config['split_train'])])
    print('Val list', list_files[int(len(list_files)*config['split_train']):int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val'])])
    print('Test list', list_files[int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val']):])

    # exit(0)

    path_images = [os.path.join(input_folder_path, im[:-4]+config['filename_ext']) for im in selected_files]
    return path_images


'''
def get_splitted_dataset(config, split, input_folder_path, path_images, path_depths, path_segmentation):
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

    ic(os.path.join(input_folder_path, config['Dataset']['paths']['path_images']))
    print('Train list', list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])])
    print('Val list', list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])])
    print('Test list', list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):])

    # exit(0)

    path_images = [os.path.join(input_folder_path, config['Dataset']['paths']['path_images'], im[:-4]+config['Dataset']['extensions']['ext_images']) for im in selected_files]
    return path_images
'''
def get_transforms(config):

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])        

    return transform_image, None, None

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_output_folders(cfg):

    path_dir_segmentation = os.path.join(cfg['path_output'], cfg['path_segmentations'])
    path_dir_uncertainty = os.path.join(cfg['path_output'], cfg['path_uncertainty'])
    path_dir_encoder_features = os.path.join(cfg['path_output'], cfg['path_encoder_features'])
    path_dir_uncertainty_map = os.path.join(cfg['path_output'], cfg['path_uncertainty_map'])

    create_dir(path_dir_segmentation)
    create_dir(path_dir_uncertainty)
    create_dir(path_dir_encoder_features)
    create_dir(path_dir_uncertainty_map)



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


def saveImages(output_segmentation, pred_entropy, filename, 
        path_dir_segmentation, path_dir_reference, path_dir_uncertainty):
    # print(np.unique(output_segmentation))

    # print(np.unique(reference_value))
    images = filename

    # output_segmentation = transforms.ToPILImage()(output_segmentation)# .resize(original_size, resample=Image.NEAREST)


    # print(os.path.join(path_dir_segmentation), os.path.basename(images))
    cv2.imwrite(os.path.join(path_dir_segmentation, os.path.basename(images)), output_segmentation*255)
    
    create_dir(path_dir_uncertainty)
    plt.imshow(pred_entropy, cmap = plt.cm.gray)
    plt.axis('off')
    plt.savefig(os.path.join(path_dir_uncertainty, os.path.basename(images)), 
        dpi=150, bbox_inches='tight', pad_inches=0.0)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
def add_margin(pil_img, padding, color):
    (top, right, bottom, left) = padding
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def pad_if_needed(im, im_size):
    if im_size[0] % 16 != 0 or im_size[1] % 16 != 0:
        padding = (0, 16 - im_size[0] % 16, 16 - im_size[1] % 16, 0)
        im = add_margin(im, padding, (128, 128, 128))
    else:
        padding = None
    return im, padding

def unpad(im, padding):
    if padding is not None:
        im = im[:-padding[2], :-padding[1]]
    return im

def save_to_csv(list, folder_path, filename):
    df = pd.DataFrame(list)
    df = df.reset_index(drop=True)
    path = pathlib.Path(folder_path)
    print("path", path)
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path / filename),
        index=False, header=False)
