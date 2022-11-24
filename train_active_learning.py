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
from icecream import ic

ic.configureOutput(includeContext=True)

with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']
list_data.append(config['ActiveLearning']['dataset'])

recommendation_idxs_path = 'recommendation_idxs_' + str(config['General']['exp_id']) + '.npy'


config_active_learning = copy.deepcopy(config)

print(recommendation_idxs_path)
recommendation_idxs = np.load(recommendation_idxs_path)
len_idxs = len(recommendation_idxs)
ic(len_idxs, int(len_idxs*config_active_learning['Dataset']['splits']['split_train']))

config_active_learning['Dataset']['splits']['split_train'] = 1.
config_active_learning['Dataset']['splits']['split_val'] = 0.
config_active_learning['Dataset']['splits']['split_test'] = 0.

if config_active_learning['ActiveLearning']['full_train'] == False:

    recommendation_idxs_train = recommendation_idxs[:int(len_idxs*0.8)]
    recommendation_idxs_validation = recommendation_idxs[int(len_idxs*0.8):]
else:
    recommendation_idxs_train = recommendation_idxs.copy()

print(config['Dataset']['splits'])

## train set
autofocus_datasets_train = []
for dataset_name in list_data:
    if dataset_name == "CorrosaoActiveLearning" or dataset_name == "CorrosaoActiveLearningReduced":
        dataset_config = config_active_learning
        
        # pdb.set_trace()
    else:
        dataset_config = config   
    print(dataset_name, dataset_config['Dataset']['splits'])     
    dataset = AutoFocusDataset(dataset_config, dataset_name, 'train')
    # pdb.set_trace()
    if dataset_name == "CorrosaoActiveLearning" or dataset_name == "CorrosaoActiveLearningReduced":

        dataset = utils.filterSamplesByIdxs(dataset, recommendation_idxs_train)
        # dataset = utils.filterSamplesByRandomIdxs(dataset, 500)

    autofocus_datasets_train.append(dataset)
train_data = ConcatDataset(autofocus_datasets_train)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True, 
    drop_last=True)

## validation set
autofocus_datasets_val = []
for dataset_name in list_data:
    if dataset_name == "CorrosaoActiveLearning" or dataset_name == "CorrosaoActiveLearningReduced":
        dataset_config = config_active_learning
        dataset = AutoFocusDataset(dataset_config, dataset_name, 'train')
        dataset = utils.filterSamplesByIdxs(dataset, recommendation_idxs_validation)
    else:
        dataset_config = config       
        dataset = AutoFocusDataset(dataset_config, dataset_name, 'val')
    autofocus_datasets_val.append(dataset)
val_data = ConcatDataset(autofocus_datasets_val)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
t0 = time.time()
trainer.train(train_dataloader, val_dataloader)
print(time.time() - t0)