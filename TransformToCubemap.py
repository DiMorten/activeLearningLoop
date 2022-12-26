from argparse import ArgumentParser

import src.cubemap as cm
import pdb
import time

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_360_images', type=str, default="C:/Users/jchamorro/Downloads/P67/P67/sample_dir2/")
parser.add_argument('-path_output', type=str, default="output/cub_maps/")
parser.add_argument('-path_cubemap_images', type=str, default="output/cub_maps_split/")
parser.add_argument('-mode', type=str, default="xprojector", choices=['xprojector', 'custom'])


args = parser.parse_args()
print(vars(args))
args = vars(args)
if __name__ == "__main__":
    if args['mode'] == 'xprojector':
        # 360 images to cubmaps, path_360_images contains all the RGB images
        cm.x_generate_cubmaps(args['path_360_images'], args['path_cubemap_images'], dims=(1344, 1344))
    elif args['mode'] == 'custom':

        # 360images to cubmaps, path_360_images contains all the RGB images
        cm.generate_cubmaps(args['path_360_images'], args['path_output'])

        # Split cubemaps into 6 images
        cm.split_cub_imgs(args['path_output'], args['path_cubemap_images'])

