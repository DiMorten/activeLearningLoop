from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="output/corrosion/")
parser.add_argument('-path_output', type=str, default="output/2D_predictions/")
# parser.add_argument('-path_output_split', type=str, default="output/cub_maps_split/")

args = parser.parse_args()
print(vars(args))
args = vars(args)

# %%
# Create the cubmap prediction (each folder contains six images)
# path_cub_prediction = root_path + 'activeLearningLoop-main/output/cub_predictions/'
path_cub_prediction = args['path_output']

if not os.path.exists(path_cub_prediction):
    os.makedirs(path_cub_prediction)
    
path_segmentation = args['path_input_cubemap_segmentation']

sub_folders = [name for name in os.listdir(path_segmentation) if os.path.isdir(os.path.join(path_segmentation, name))]
for i in range(0, len(sub_folders)):
    print('image: ', i)
    cube_prediction = cm.join_images(path_segmentation + sub_folders[i] + '/segmentations/')
    imageio.imwrite(path_cub_prediction + sub_folders[i] +'.png', cube_prediction)