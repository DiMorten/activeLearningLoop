import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from FOD.Trainer import Trainer
from FOD.dataset import AutoFocusDataset

import time
import pdb
import copy

import FOD.utils as utils


with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']


config_active_learning = copy.deepcopy(config)

if config_active_learning['ActiveLearning']['full_train'] == False:
    config_active_learning['Dataset']['splits']['split_train'] = 0.8
    config_active_learning['Dataset']['splits']['split_val'] = 0.2
    config_active_learning['Dataset']['splits']['split_test'] = 0.
else:
    config_active_learning['Dataset']['splits']['split_train'] = 1.
    config_active_learning['Dataset']['splits']['split_val'] = 0.
    config_active_learning['Dataset']['splits']['split_test'] = 0.

print(config['Dataset']['splits'])

## train set
autofocus_datasets_train = []
for dataset_name in list_data:
    if dataset_name == "CorrosaoActiveLearning":
        dataset_config = config_active_learning
        
        # pdb.set_trace()
    else:
        dataset_config = config   
    print(dataset_name, dataset_config['Dataset']['splits'])     
    dataset = AutoFocusDataset(dataset_config, dataset_name, 'train')

    if dataset_name == "CorrosaoActiveLearning":
        recommendation_idxs = np.load('recommendation_idxs.npy')
        dataset = utils.filterSamplesByIdxs(dataset, recommendation_idxs)

    autofocus_datasets_train.append(dataset)
train_data = ConcatDataset(autofocus_datasets_train)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True, 
    drop_last=True)

## validation set
autofocus_datasets_val = []
for dataset_name in list_data:
    autofocus_datasets_val.append(AutoFocusDataset(config, dataset_name, 'val'))
val_data = ConcatDataset(autofocus_datasets_val)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
t0 = time.time()
trainer.train(train_dataloader, val_dataloader)
print(time.time() - t0)