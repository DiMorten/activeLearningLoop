from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="output/corrosion/")
parser.add_argument('-path_output_cubemap', type=str, default="output/2D_predictions/")
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
path_cub_prediction = args['path_output_cubemap']

if not os.path.exists(path_cub_prediction):
    os.makedirs(path_cub_prediction)
    
path_segmentation = args['path_input_cubemap_segmentation']

df = pd.read_csv(args['path_csv'], header=None)
print(df)

for i in range(0, len(df)):
    print('image: ', i)
    cube_prediction = cm.join_images_from_name(path_segmentation + args['cubemap_keyword'] + '_' + df[0][i])
    imageio.imwrite(path_cub_prediction + df[0][i] +'.png', cube_prediction)    
# pdb.set_trace()


# %%
print("...Finished cubemap to 2D conversion")
print("Time:", t0 - time.time())
print("Starting 2D to 360 conversion...")
path_cub_pred = args['path_output_cubemap']
path_360_pred = args['path_output_360']
if not os.path.exists(path_360_pred):
    os.makedirs(path_360_pred)
        
img_pred = cm.return_files(path_cub_pred)
print(img_pred)

# Transform each cubmap prediction into a 360 image prediction
for i in range(0, len(img_pred)):
    print(i)
    cm.convert_img(path_cub_pred + img_pred[i], path_360_pred + img_pred[i])

print("...Finished 2D to 360 conversion")
print("Time:", t0 - time.time())
