from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd
from pathlib import Path

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemaps', type=str, default="output/cub_maps_split/")
parser.add_argument('-path_output_2D', type=str, default="output/2D_predictions/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")

'''
parser.add_argument('-path_input_cubemaps', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/cub_maps_split/")
parser.add_argument('-path_output_2D', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/2D_predictions/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
'''

args = parser.parse_args()

args = vars(args)



if __name__ == "__main__":
    if args['mode'] == 'xprojector':
        # %%
        print("Starting cubemap to 2D conversion...")

        # Create output folder
        if not os.path.exists(args['path_output_2D']):
            os.makedirs(args['path_output_2D'])

        # Get cubemap input files
        filenames = [os.path.basename(str(i)) for i in Path(args['path_input_cubemaps']).glob('*.png')]

        filenames_360 = cm.get_unique_from_cubemaps2(filenames)
        filenames_360 = list(set(filenames_360))

        print('total of {} input files'.format(len(filenames_360)))

        # ignore already processed files
        filenames_360 = cm.ignore_already_processed_cubemaps(filenames_360, args['path_output_2D'])


        print('number of pending files: {}'.format(len(filenames_360)))


        cm.cubemaps_to_2D(args['path_input_cubemaps'], args['cubemap_keyword'], 
                filenames_360, args['path_output_2D'])
        