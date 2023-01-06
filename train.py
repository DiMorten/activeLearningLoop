import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from src.Trainer import Trainer
from src.dataset import AutoFocusDataset

import time
import pdb
import copy

import src.utils as utils
from icecream import ic
import argparse
ic.configureOutput(includeContext=True)

app_desc = ''
parser = argparse.ArgumentParser(
    description = app_desc, 
    formatter_class = argparse.RawDescriptionHelpFormatter,
    allow_abbrev=True)

parser.add_argument('-filename', type=str)

args = parser.parse_args()
print(args)


with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

if args.filename is not None:
    list_data = [args.filename]
else:
    list_data = [config['Dataset']['paths']['path_dataset'] + '/' + x for x in config['Dataset']['paths']['list_datasets']]

# list_data = config['Dataset']['paths']['list_datasets']
# list_data = [config['Dataset']['paths']['path_dataset'] + '/' + x for x in config['Dataset']['paths']['list_datasets']]
print(list_data)
# pdb.set_trace()
config['General']['load_reference'] = True
config['Dataset']['splits']['split_train'] = 0.75
config['Dataset']['splits']['split_val'] = 0.25
config['Dataset']['splits']['split_test'] = 0.

print(config['Dataset']['splits'])

## train set
autofocus_datasets_train = []
for dataset_name in list_data:

    dataset_config = config   
    print(dataset_name, dataset_config['Dataset']['splits'])     
    dataset = AutoFocusDataset(dataset_config, dataset_name, 'train')
    autofocus_datasets_train.append(dataset)
train_data = ConcatDataset(autofocus_datasets_train)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True, 
    drop_last=True)

## validation set
autofocus_datasets_val = []
for dataset_name in list_data:

    dataset_config = config       
    autofocus_datasets_val.append(AutoFocusDataset(dataset_config, dataset_name, 'val'))
val_data = ConcatDataset(autofocus_datasets_val)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
t0 = time.time()
trainer.train(train_dataloader, val_dataloader)
print(time.time() - t0)