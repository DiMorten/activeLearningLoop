import os
import random
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.utils import get_total_paths, get_splitted_dataset, get_transforms, ignore_already_computed
import pdb

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu').float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class HilaiDataset(Dataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, its depth ground-truth and
        segmentation mask
        Args:
            :- config -: json config file
            :- input_folder_path -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config):

        self.config = config


        self.split = config['split']
        input_folder_path = config['filename']
        self.use_reference = config['use_reference']

        path_images = input_folder_path # config['path_images']

        self.paths_images = get_total_paths(path_images, config['filename_ext'])
        # print(self.paths_images)
        # pdb.set_trace()
        # self.paths_images = ignore_already_computed(self.paths_images)
        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        print(len(self.paths_images))


        if self.use_reference == True:
            assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
        assert (config['split_train']+config['split_test']+config['split_val'] == 1), "Invalid splits (sum must be equal to 1)"
        # check for segmentation

        # ic(input_folder_path, self.paths_images)
        # utility func for splitting

        self.paths_images = get_splitted_dataset(config, self.split, input_folder_path, self.paths_images)

        # Get the transforms
        self.transform_image, self.transform_depth, self.transform_seg = get_transforms(config)

        # get p_flip from config
        self.p_flip = 0
        self.p_crop = 0
        self.p_rot = 0
        self.resize = 512

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images / depth maps and segmentation masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform_image(Image.open(self.paths_images[idx]))
        # ic(Image.open(self.paths_depths[idx]).mode)
        # ic(Image.open(self.paths_segmentations[idx]).mode)
        # depth = self.transform_depth(Image.open(self.paths_depths[idx]))
        if self.use_reference == True:
            segmentation = self.transform_seg(Image.open(self.paths_segmentations[idx]))
        # imgorig = image.clone()

        if random.random() < self.p_flip:
            image = TF.hflip(image)
            # depth = TF.hflip(depth)
            if self.use_reference == True:
                segmentation = TF.hflip(segmentation)

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.resize-1)
            max_size = self.resize - random_size
            left = int(random.random()*max_size)
            top = int(random.random()*max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            # depth = TF.crop(depth, top, left, random_size, random_size)
            if self.use_reference == True:
                segmentation = TF.crop(segmentation, top, left, random_size, random_size)
            image = transforms.Resize((self.resize, self.resize))(image)
            # depth = transforms.Resize((self.resize, self.resize))(depth)
            if self.use_reference == True:
                segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

        if random.random() < self.p_rot:
            #rotate
            random_angle = random.random()*20 - 10 #[-10 ; 10]
            mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            # depth = TF.rotate(depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            if self.use_reference == True:
                segmentation = TF.rotate(segmentation, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
            #crop to remove black borders due to the rotation
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left,top)
            size = self.resize - 2*coin
            image = TF.crop(image, coin, coin, size, size)
            # depth = TF.crop(depth, coin, coin, size, size)
            if self.use_reference == True:
                segmentation = TF.crop(segmentation, coin, coin, size, size)
            #Resize
            image = transforms.Resize((self.resize, self.resize))(image)
            #depth = transforms.Resize((self.resize, self.resize))(depth)
            if self.use_reference == True:
                segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)
        # show([imgorig, image, depth, segmentation])
        # exit(0)
        if self.use_reference == True:
            return image, segmentation, self.paths_images[idx]
        else:
            return image, self.paths_images[idx]


