import json
from glob import glob
from FOD.Predictor import PredictorMCDropout, PredictorEntropyAL
import copy
import argparse
import os

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['Dataset']['paths']['path_dataset'], config['ActiveLearning']['dataset'])

predictor = PredictorEntropyAL(config, input_folder_path)
predictor.run()
