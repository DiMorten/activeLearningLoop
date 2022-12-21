from argparse import ArgumentParser

import src.cubemap as cm
import pdb
import time

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input', type=str, default="C:/Users/jchamorro/Downloads/P67/P67/sample_dir2/imgs/")
parser.add_argument('-path_output', type=str, default="output/cub_maps/")
parser.add_argument('-path_output_split', type=str, default="output/cub_maps_split/")


args = parser.parse_args()
print(vars(args))
args = vars(args)

# 360 images to cubmaps, path_input contains all the RGB images
cm.generate_cubmaps(args['path_input'], args['path_output'])

# Split cubemaps into 6 images
cm.split_cub_imgs(args['path_output'], args['path_output_split'])

# init the model
