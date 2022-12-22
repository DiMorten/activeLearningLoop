from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="output/corrosion/")
parser.add_argument('-path_output_2D', type=str, default="output/2D_predictions/")
parser.add_argument('-path_csv', type=str, default="output/inference_csv.csv")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="output/corrosion_360/")

# parser.add_argument('-path_output_split', type=str, default="output/cub_maps_split/")


args = parser.parse_args()
print(vars(args))
args = vars(args)

print("Starting cubemap to 2D conversion...")
t0 = time.time()
# %%
# Create the cubmap prediction (each folder contains six images)
# path_cub_prediction = root_path + 'activeLearningLoop-main/output/cub_predictions/'

if not os.path.exists(args['path_output_2D']):
    os.makedirs(args['path_output_2D'])
    
# Read CSV with list of inference 360 images
df = pd.read_csv(args['path_csv'], header=None)
print(df)

# Transform cubemap faces to 2D cubemap representation
for i in range(0, len(df)):
    print('image: ', i)

    cm.cubemap_to_360(args['path_input_cubemap_segmentation'], args['cubemap_keyword'], 
        df[0][i], args['path_output_360'])
        
    
print("...Finished cubemap to 2D representation conversion. Time:", t0 - time.time())

# %%
'''
print("Starting 2D to 360 conversion...")

if not os.path.exists(args['path_output_360']):
    os.makedirs(args['path_output_360'])
        
img_pred = cm.return_files(args['path_output_2D'])
print(img_pred)

# Transform each cubmap prediction into a 360 image prediction
for i in range(0, len(img_pred)):
    print(i)
    cm.convert_img(args['path_output_2D'] + img_pred[i], args['path_output_360'] + img_pred[i])

print("...Finished 2D to 360 conversion. Time:", t0 - time.time())
'''