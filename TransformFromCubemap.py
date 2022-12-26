from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd
from pathlib import Path

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="output/corrosion/")
parser.add_argument('-path_output_2D', type=str, default="output/2D_predictions/")
parser.add_argument('-path_csv', type=str, default="output/inference_csv.csv")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="output/corrosion_360/")
parser.add_argument('-mode', type=str, default="xprojector", choices=['xprojector', 'custom'])
parser.add_argument('-n_jobs', type=int, default=1)
'''
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/corrosion/")
parser.add_argument('-path_output_2D', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/2D_predictions/")
parser.add_argument('-path_csv', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/inference_csv.csv")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/corrosion_360/")
parser.add_argument('-mode', type=str, default="xprojector", choices=['xprojector', 'custom'])
parser.add_argument('-n_jobs', type=int, default=1)
'''
# parser.add_argument('-path_output_split', type=str, default="output/cub_maps_split/")


args = parser.parse_args()
print(vars(args))
args = vars(args)

# Read CSV with list of inference 360 images
files = ['_'.join(str(i).split('/')[-1].split('_')[1:4]) for i in Path(args['path_input_cubemap_segmentation']).glob('**/*.png')]

df = files #pd.read_csv(args['path_csv'], header=None)


t0 = time.time()

def init_folders(args):
    if not os.path.exists(args['path_output_2D']):
        os.makedirs(args['path_output_2D'])

    if not os.path.exists(args['path_output_360']):
        os.makedirs(args['path_output_360'])


    with open(os.path.join(
        os.path.dirname(args['path_csv']),"unsuccessful_from_cubemap.txt"), 'w') as f:
        f.write('')

if __name__ == "__main__":
    if args['mode'] == 'xprojector':
        # %%
        print("Starting cubemap to 360 conversion...")
        init_folders(args)
        # Create the cubmap prediction (each folder contains six images)
        # path_cub_prediction = root_path + 'activeLearningLoop-main/output/cub_predictions/'



        filenames_360 = []
        for i in range(0, len(df)):
            filenames_360.append(df[i])

        cm.cubemaps_to_360(args['path_input_cubemap_segmentation'], args['cubemap_keyword'], 
            filenames_360, args['path_output_360'], args['path_csv'], n_jobs=args['n_jobs'])
                

            
        print("...Finished cubemap to 360 conversion. Time:", time.time() - t0)
    elif args['mode'] == 'custom':

        print("Starting cubemap to 2D conversion...")

        # %%
        # Create the cubmap prediction (each folder contains six images)
        # path_cub_prediction = root_path + 'activeLearningLoop-main/output/cub_predictions/'

        if not os.path.exists(args['path_output_2D']):
            os.makedirs(args['path_output_2D'])
            
        for i in range(0, len(df)):
            print('image: ', i)
            cm.cubemap_to_2D(args['path_input_cubemap_segmentation'], args['cubemap_keyword'], 
                df[i], args['path_output_2D'])
                
        print("...Finished cubemap to 2D conversion. Time:", t0 - time.time())

        # %%

        print("Starting 2D to 360 conversion...")

        if not os.path.exists(args['path_output_360']):
            os.makedirs(args['path_output_360'])
                
        img_pred = cm.return_files(args['path_output_2D'])
        print(img_pred)

        # Transform each cubmap prediction into a 360 image prediction
        for i in range(0, len(img_pred)):
            print(i)
            print(args['path_output_2D'] + img_pred[i], args['path_output_360'] + img_pred[i])
            cm.convert_img(args['path_output_2D'] + img_pred[i], args['path_output_360'] + img_pred[i])
            

        print("...Finished 2D to 360 conversion. Time:", t0 - time.time())

