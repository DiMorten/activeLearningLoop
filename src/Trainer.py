import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
import torch.nn as nn
from torchsummary import summary

from tqdm import tqdm
from os import replace
from numpy.core.numeric import Inf
from src.utils import get_losses, get_optimizer, get_schedulers, create_dir
from src.FocusOnDepth import FocusOnDepth
from src.FCNs import ResUnetPlusPlus

import sys
sys.path.append('E:/jorg/phd/visionTransformer/activeLearningLoop/segmentation_models_ptorch')

import segmentation_models_pytorch as smp
import segmentation_models_pytorch_dropout as smpd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn.functional as F
import pdb

class EarlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.restartCounter()
    def restartCounter(self):
        self.counter = 0
    def increaseCounter(self):
        self.counter += 1
    def checkStopping(self):
        if self.counter >= self.patience:
            return True
        else:
            return False
class Trainer(object):
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        if config['General']['model_type'] == 'FocusOnDepth':
            self.model = FocusOnDepth(
                        image_size  =   (3,resize,resize),
                        emb_dim     =   config['General']['emb_dim'],
                        resample_dim=   config['General']['resample_dim'],
                        read        =   config['General']['read'],
                        nclasses    =   len(config['Dataset']['classes']) + 1,
                        hooks       =   config['General']['hooks'],
                        model_timm  =   config['General']['model_timm'],
                        type        =   self.type,
                        patch_size  =   config['General']['patch_size'],
            )

        elif config['General']['model_type'] == 'unet':        
            
            self.model = smp.Unet('xception', encoder_weights='imagenet', in_channels=3,
                encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=2)
        elif config['General']['model_type'] == 'deeplab':        
            
            self.model = smp.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', in_channels=3,
                classes=2)

        elif config['General']['model_type'] == 'deeplab_dropout':        
            
            self.model = smpd.DeepLabV3Plus('resnet34', encoder_weights='imagenet', in_channels=3,
                classes=2)

            # pdb.set_trace()

                
        '''
        self.model = ResUnetPlusPlus(
                    image_size  =   (3,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )
        '''

        '''
        aux_params=dict(
            dropout=0.5,               # dropout ratio, default is None
            classes=2
        )
        '''

        # self.model = smp.DeepLabV3Plus('xception', encoder_weights='imagenet', in_channels=3, 
        #     encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=2)

        self.model.to(self.device)
        '''
        if config['General']['model_type'] == 'deeplab':   
            dropout_ = DropoutHook(prob=0.2)
            # self.model.apply(dropout_.register_hook)

            print(self.model.encoder.model.blocks_1.stack)

            # self.model.encoder.model.blocks_1.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_4.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_7.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_10.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_12.apply(dropout_.register_hook)
        '''
        # print(self.model)

        # print(self.model.encoder.model.blocks_1.stack)

        # pdb.set_trace()
        # exit(0)
        # print("input shape: ", (3,resize,resize))
        # print(resize)
        # summary(self.model, (3,resize,resize))
        # exit(0)

        self.loss_depth, self.loss_segmentation = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])

        self.path_model = os.path.join(self.config['General']['path_model'], 
            self.model.__class__.__name__ + 
            '_' + str(self.config['General']['exp_id']))

            
    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']
        if self.config['wandb']['enable']:
            wandb.init(project="FocusOnDepth", entity=self.config['wandb']['username'])
            wandb.config = {
                "learning_rate_backbone": self.config['General']['lr_backbone'],
                "learning_rate_scratch": self.config['General']['lr_scratch'],
                "epochs": epochs,
                "batch_size": self.config['General']['batch_size']
            }
        val_loss = Inf
        es = EarlyStopping(10)
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch ", epoch+1)
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            
            for i, (X, Y_segmentations) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                X, Y_segmentations = X.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients
                if isinstance(self.model, FocusOnDepth):
                    self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer
                if isinstance(self.model, FocusOnDepth):
                    output_depths, output_segmentations = self.model(X)
                else:
                    output_depths, output_segmentations = (None, self.model(X))
                
                output_depths = output_depths.squeeze(1) if output_depths != None else None

                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()
                if isinstance(self.model, FocusOnDepth):
                    self.optimizer_backbone.step()

                running_loss += loss.item()
                if np.isnan(running_loss):
                    print('\n',
                        X.min().item(), X.max().item(),'\n',
                        loss.item(),
                    )
                    exit(0)

                if self.config['wandb']['enable'] and ((i % 50 == 0 and i>0) or i==len(train_dataloader)-1):
                    wandb.log({"loss": running_loss/(i+1)})
                pbar.set_postfix({'training_loss': running_loss/(i+1)})

            new_val_loss = self.run_eval(val_dataloader)

            if new_val_loss < val_loss:
                self.save_model()
                val_loss = new_val_loss
                es.restartCounter()
            else:
                es.increaseCounter()

            self.schedulers[0].step(new_val_loss)
            if isinstance(self.model, FocusOnDepth):
                self.schedulers[1].step(new_val_loss)

            if es.checkStopping() == True:
                print("Early stopping")
                print(es.counter, es.patience)
                print('Finished Training')
                exit(0)

        print('Finished Training')

    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.
        self.model.eval()
        X_1 = None
        Y_segmentations_1 = None
        output_segmentations_1 = None
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_segmentations) in enumerate(pbar):
                X, Y_segmentations = X.to(self.device), Y_segmentations.to(self.device)
                if isinstance(self.model, FocusOnDepth):
                    output_depths, output_segmentations = self.model(X)
                else:
                    output_depths, output_segmentations = (None, self.model(X))

                Y_segmentations = Y_segmentations.squeeze(1)
                if i==0:
                    X_1 = X
                    Y_segmentations_1 = Y_segmentations
                    output_segmentations_1 = output_segmentations
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                val_loss += loss.item()
                pbar.set_postfix({'validation_loss': val_loss/(i+1)})
            # if self.config['wandb']['enable']:
            #     wandb.log({"val_loss": val_loss/(i+1)})
            #     self.img_logger(X_1, Y_depths_1, Y_segmentations_1, output_depths_1, output_segmentations_1)
        return val_loss/(i+1)

    def save_model(self):
        
        create_dir(self.path_model)
        torch.save({'model_state_dict': self.model.state_dict(),
                    # 'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                    'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                    }, self.path_model+'.p')
        print('Model saved at : {}'.format(self.path_model))

    def img_logger(self, X, Y_depths, Y_segmentations, output_depths, output_segmentations):
        nb_to_show = self.config['wandb']['images_to_show'] if self.config['wandb']['images_to_show'] <= len(X) else len(X)
        tmp = X[:nb_to_show].detach().cpu().numpy()
        imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        if output_depths != None:
            tmp = Y_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            depth_truths = np.repeat(tmp, 3, axis=1)
            tmp = output_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            #depth_preds = 1.0 - tmp
            depth_preds = tmp
        if output_segmentations != None:
            tmp = Y_segmentations[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            segmentation_truths = np.repeat(tmp, 3, axis=1).astype('float32')
            tmp = torch.argmax(output_segmentations[:nb_to_show], dim=1)
            tmp = tmp.unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            segmentation_preds = tmp.astype('float32')
        # print("******************************************************")
        # print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
        # if output_depths != None:
        #     print(depth_truths.shape, depth_truths.mean().item(), depth_truths.max().item(), depth_truths.min().item())
        #     print(depth_preds.shape, depth_preds.mean().item(), depth_preds.max().item(), depth_preds.min().item())
        # if output_segmentations != None:
        #     print(segmentation_truths.shape, segmentation_truths.mean().item(), segmentation_truths.max().item(), segmentation_truths.min().item())
        #     print(segmentation_preds.shape, segmentation_preds.mean().item(), segmentation_preds.max().item(), segmentation_preds.min().item())
        # print("******************************************************")
        imgs = imgs.transpose(0,2,3,1)
        if output_depths != None:
            depth_truths = depth_truths.transpose(0,2,3,1)
            depth_preds = depth_preds.transpose(0,2,3,1)
        if output_segmentations != None:
            segmentation_truths = segmentation_truths.transpose(0,2,3,1)
            segmentation_preds = segmentation_preds.transpose(0,2,3,1)
        output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))

        wandb.log({
            "img": [wandb.Image(cv2.resize(im, output_dim), caption='img_{}'.format(i+1)) for i, im in enumerate(imgs)]
        })
        if output_depths != None:
            wandb.log({
                "depth_truths": [wandb.Image(cv2.resize(im, output_dim), caption='depth_truths_{}'.format(i+1)) for i, im in enumerate(depth_truths)],
                "depth_preds": [wandb.Image(cv2.resize(im, output_dim), caption='depth_preds_{}'.format(i+1)) for i, im in enumerate(depth_preds)]
            })
        if output_segmentations != None:
            wandb.log({
                "seg_truths": [wandb.Image(cv2.resize(im, output_dim), caption='seg_truths_{}'.format(i+1)) for i, im in enumerate(segmentation_truths)],
                "seg_preds": [wandb.Image(cv2.resize(im, output_dim), caption='seg_preds_{}'.format(i+1)) for i, im in enumerate(segmentation_preds)]
            })
