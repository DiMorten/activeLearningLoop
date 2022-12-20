import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from FOD.dataset import HilaiDataset
from glob import glob
from argparse import ArgumentParser
import pdb
from pytorch_lightning.callbacks import Callback
import numpy as np
import sys
sys.path.append('E:/jorg/phd/visionTransformer/activeLearningLoop/segmentation_models_ptorch')
import segmentation_models_pytorch_dropout as smpd
import json
from FOD.uncertainty import get_uncertainty_map
import os
from FOD.utils import create_dir, create_output_folders
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
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


        
        self.model = smpd.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', 
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

        # parser.add_argument("--encoder_weights", type=str, default=12)
        # parser.add_argument("--data_path", type=str, default="/some/path")
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
parser.add_argument('-filename', type=str, default="input/")

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

parser.add_argument('-test_batch_size', type=int, default=1)

parser.add_argument('-path_segmentations', type=str, default='corrosion')
parser.add_argument('-path_uncertainty', type=str, default='uncertainty')
parser.add_argument('-path_uncertainty_map', type=str, default='uncertainty_map')
parser.add_argument('-path_encoder_features', type=str, default='encoder_features')

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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        '''
        print(outputs['softmax'].shape)
        print(outputs['segmentations'].shape)
        print(outputs['uncertainty_map'].shape)
        print(outputs['uncertainty'].shape)
        print(outputs['encoder_features'].shape)
        '''
        # print(outputs['filenames'])
        # pdb.set_trace()
        x, filenames = batch

        for idx in range(x.shape[0]):
            filename = filenames[idx].split('/')[-1]
            np.savez(args['path_output'] +'/'+ args['path_encoder_features'] +'/'+ filename[:-4] + '.npz', 
                outputs['encoder_features'].cpu().detach().numpy()[idx])
            # np.savez(args['path_output'] +'/'+ args['path_uncertainty'] +'/'+ filename[:-4] + '.npz', 
            #     outputs['uncertainty'][idx])
            # np.savez(args['path_output'] +'/'+ args['path_segmentations'] +'/'+ filename + '.npz', 
            #     outputs['softmax'][idx])
            # pdb.set_trace()
            # print(os.path.join(args['path_output'], args['path_segmentations'], filename))
            cv2.imwrite(os.path.join(args['path_output'], args['path_segmentations'], filename), outputs['segmentations'][idx]*255)

            
            plt.imshow(outputs['uncertainty_map'][idx], cmap = plt.cm.gray)
            plt.axis('off')
            plt.savefig(args['path_output'] +'/'+ args['path_uncertainty_map'] +'/'+ filename, 
                dpi=150, bbox_inches='tight', pad_inches=0.0)
        # pdb.set_trace()

        # salvar mascara de softmax e mapa de incerteza
        # pl_module
        # outputs

        # batch: passar
        pass

trainer = pl.Trainer.from_argparse_args(args, callbacks=[SaveOutcomesCallback()],
    gpus=-1)

args = vars(args)

create_output_folders(args)

# pdb.set_trace()
# init the model
model = LitModel(**args)

dm = HilaiDataModule(**args)
dm.setup("validation")
val_dataloader = dm.val_dataloader()
trainer.validate(model=model, dataloaders=val_dataloader)
 

