import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.dataset import HilaiDataset
from glob import glob
from argparse import ArgumentParser
import pdb
from pytorch_lightning.callbacks import Callback
import numpy as np
import sys
sys.path.append('segmentation_models_ptorch')
import segmentation_models_pytorch_custom as smpc
from src.uncertainty import get_uncertainty_map
import os
from src.utils import create_dir, create_output_folders, save_to_csv
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
# define the LightningModule
class LitModel(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # fill arguments
        
        parser.add_argument('-filename', type=str, default="input/")
        parser.add_argument('-filename_ext', type=str, default=".png")
        parser.add_argument('-path_images', type=str, default="imgs")

        parser.add_argument('-split_train', type=float, default=0.)
        parser.add_argument('-split_val', type=float, default=0.)
        parser.add_argument('-split_test', type=float, default=1.)

        parser.add_argument('-split', type=str, default='test')
        parser.add_argument('-use_reference', type=bool, default=False)
        parser.add_argument('-seed', type=int, default=0)

        parser.add_argument('-test_batch_size', type=int, default=1)

        return parent_parser
        
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)


        
        self.model = smpc.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', 
                    in_channels=3, classes=2)

        path_model = os.path.join(cfg['path_model'], self.model.__class__.__name__ + 
            '_' + str(cfg['exp_id']) + '.p')
        print(path_model)
        # pdb.set_trace()
        self.model.load_state_dict(
            torch.load(path_model)['model_state_dict']
        )

        self.transform_image = transforms.Compose([   
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])                  

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        x, filenames = batch

        encoder_features, y = self.model(x)
        encoder_features = encoder_features.mean((2, 3))
        y = torch.nn.functional.softmax(y, dim=1)
        y = y.cpu().detach().numpy()
        segmentations = np.argmax(y, axis=1).astype(np.uint8)

        y = y[:, 1]
        # Use metrics module to calculate uncertainty metric
        uncertainty_map = get_uncertainty_map(np.expand_dims(y, axis=-1))
        
        uncertainty = np.mean(uncertainty_map, axis=(1, 2))   
        # Logging to TensorBoard by default
        # self.log("train_loss", loss)
        
        return {'softmax': y, 'segmentations': segmentations, 
            'uncertainty_map': uncertainty_map, 'uncertainty': uncertainty,
            'encoder_features': encoder_features, 'filenames': filenames}
        # return y
class HilaiDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HilaiDataModule")
        # fill arguments

        parser.add_argument('-filename', type=str, default="input/")
        parser.add_argument('-filename_ext', type=str, default=".png")
        parser.add_argument('-path_images', type=str, default="imgs")

        parser.add_argument('-split_train', type=float, default=0.)
        parser.add_argument('-split_val', type=float, default=0.)
        parser.add_argument('-split_test', type=float, default=1.)

        parser.add_argument('-split', type=str, default='test')
        parser.add_argument('-use_reference', type=bool, default=False)
        parser.add_argument('-seed', type=int, default=0)

        parser.add_argument('-test_batch_size', type=int, default=2)


        return parent_parser
        
    def __init__(self, **cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        
        # self.data_dir = data_dir
        # self.batch_size = batch_size
    def setup(self, stage: str):
        self.dataset_val = HilaiDataset(self.cfg)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg['test_batch_size'])


parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-filename', type=str, default="output/cub_maps_split")

parser.add_argument('-filename_ext', type=str, default=".png")

parser.add_argument('-path_output', type=str, default="output")
parser.add_argument('-path_images', type=str, default="imgs")
parser.add_argument('-path_model', type=str, default="models")
parser.add_argument('-exp_id', type=int, default=0)


parser.add_argument('-split_train', type=float, default=0.)
parser.add_argument('-split_val', type=float, default=0.)
parser.add_argument('-split_test', type=float, default=1.)

parser.add_argument('-split', type=str, default='test')
parser.add_argument('-use_reference', type=bool, default=False)
parser.add_argument('-seed', type=int, default=0)

parser.add_argument('-test_batch_size', type=int, default=6)

parser.add_argument('-path_segmentations', type=str, default='corrosion')
parser.add_argument('-path_uncertainty', type=str, default='uncertainty')
parser.add_argument('-path_uncertainty_map', type=str, default='uncertainty_map')
parser.add_argument('-path_encoder_features', type=str, default='encoder_features')

parser.add_argument('-test_csv_name', type=str, default='inference_csv')
parser.add_argument('-mean_uncertainty_csv_name', type=str, default='mean_uncertainty')

# add model specific args
## parser = LitModel.add_model_specific_args(parser)
## parser = HilaiDataModule.add_model_specific_args(parser)
# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
## parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# pdb.set_trace()
print(vars(args))

class SaveOutcomesCallback(Callback):
    def on_validation_start(self, trainer, pl_module):
        self.validation_filenames = []
        self.uncertainty_mean_values = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        '''
        print(outputs['softmax'].shape)
        print(outputs['segmentations'].shape)
        print(outputs['uncertainty_map'].shape)
        print(outputs['uncertainty'].shape)
        print(outputs['encoder_features'].shape)
        print(outputs['filenames'])
        '''
        # 
        # pdb.set_trace()
        x, filenames = batch

        for idx in range(x.shape[0]):
            filename = filenames[idx].split('/')[-1].split('\\')[-1]
            self.validation_filenames.append(filename)
            self.uncertainty_mean_values.append(outputs['uncertainty'][idx])

            np.savez(args['path_output'] +'/'+ args['path_encoder_features'] +'/'+ filename.split('.')[0] + '.npz', 
                outputs['encoder_features'].cpu().detach().numpy()[idx])
            np.savez(args['path_output'] +'/'+ args['path_uncertainty_map'] +'/'+ filename.split('.')[0] + '.npz', 
                outputs['uncertainty_map'][idx])
            
            cv2.imwrite(os.path.join(args['path_output'], args['path_segmentations'], filename), outputs['segmentations'][idx]*255)

            
    def on_validation_end(self, trainer, pl_module):

        # Save CSV with 360 image names
        self.validation_filenames_360 = [x.split('.')[0].split('_')[-2] for x in self.validation_filenames]
        self.validation_filenames_360 = list(dict.fromkeys(self.validation_filenames_360))
        print(self.validation_filenames_360)

        save_to_csv(self.validation_filenames_360, 
            args['path_output'],
            args['test_csv_name'] + '.csv')
        

        # Save CSV with mean uncertainty

        save_to_csv(zip(self.validation_filenames, self.uncertainty_mean_values), 
            args['path_output'],
            args['mean_uncertainty_csv_name'] + '.csv')


trainer = pl.Trainer.from_argparse_args(args, callbacks=[SaveOutcomesCallback()],
    gpus=-1)

args = vars(args)

create_output_folders(args)

# init the model
model = LitModel(**args)

dm = HilaiDataModule(**args)
dm.setup("validation")
val_dataloader = dm.val_dataloader()
trainer.validate(model=model, dataloaders=val_dataloader)
 

