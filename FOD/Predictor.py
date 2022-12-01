import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from icecream import ic
from PIL import Image
import pdb
from scipy.special import softmax

from FOD.FocusOnDepth import FocusOnDepth

from FOD.utils import create_dir
from FOD.dataset import show
import FOD.uncertainty as uncertainty

import sys, pdb
sys.path.append('E:/jorg/phd/visionTransformer/activeLearningLoop/segmentation_models_ptorch')

import segmentation_models_pytorch as smp
import segmentation_models_pytorch_dropout as smpd

import time

from sklearn import metrics
import FOD.ActiveLearning as al
from FOD.ActiveLearning import ActiveLearner
from icecream import ic
import copy
import FOD.utils as utils

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model
def check_dropout_enabled(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print(m.training)
    return model

class Predictor(object):
    def __init__(self, config, input_images):
        self.input_images = input_images
        self.config = config
        self.type = self.config['General']['type']
        self.save_images = self.config['Inference']['save_images']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        # resize = 513
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
            # path_model = os.path.join(config['General']['path_model'], 'FocusOnDepth_{}.p'.format(config['General']['model_timm']))

        elif config['General']['model_type'] == 'unet':        
            
            self.model = smp.Unet('xception', encoder_weights='imagenet', in_channels=3,
                encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=2)
            # path_model = os.path.join(config['General']['path_model'], 'Unet.p')


        elif config['General']['model_type'] == 'deeplab':        
            if config['ActiveLearning']['diversity_method'] != False:
                network = smpd.DeepLabV3Plus
            else:
                network = smp.DeepLabV3Plus   
            
            self.model = network('tu-xception41', encoder_weights='imagenet', in_channels=3,
                    classes=2)                
            # path_model = os.path.join(config['General']['path_model'], 'DeepLabV3Plus.p')

        elif config['General']['model_type'] == 'deeplab_dropout':        
            
            self.model = smpd.DeepLabV3Plus('resnet34', encoder_weights='imagenet', in_channels=3,
                classes=2)
            # path_model = os.path.join(config['General']['path_model'], 'DeepLabV3Plus.p')

        
        path_model = os.path.join(self.config['General']['path_model'], self.model.__class__.__name__ + 
            '_' + str(self.config['General']['exp_id']) + '.p')
        ic(path_model)
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

        # path_model = os.path.join(config['General']['path_model'], 'ResUnetPlusPlus.p')
        
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.eval()

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
        self.transform_image = transforms.Compose([
            transforms.Resize((resize, resize)),
            # transforms.Resize((528, 528)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.output_dir = self.config['General']['path_predicted_images']
        create_dir(self.output_dir)

    def run(self):
        with torch.no_grad():
            for images in self.input_images:
                pil_im = Image.open(images)
                original_size = pil_im.size

                tensor_im = self.transform_image(pil_im).unsqueeze(0)
                if isinstance(self.model, FocusOnDepth):
                    _, output_segmentation = self.model(tensor_im)
                else:
                    _, output_segmentation = (None, self.model(tensor_im))

                # output_depth = 1-output_depth
                if self.save_images == True:
                    output_segmentation = transforms.ToPILImage()(output_segmentation.squeeze(0).argmax(dim=0).float()).resize(original_size, resample=Image.NEAREST)
                    # output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)

                    path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')
                    path_dir_depths = os.path.join(self.output_dir, 'depths')
                    create_dir(path_dir_segmentation)
                    output_segmentation.save(os.path.join(path_dir_segmentation, os.path.basename(images)))


    def inferDataLoader(self, dataloader, getEncoder = False):
        pbar = tqdm(dataloader)
        pbar.set_description("Testing")
        self.model.to(self.device)

        softmax_segmentations = []
        output_values = []
        uncertainty_values = []
        reference_values = []
        encoder_values = []

        if self.config['ActiveLearning']['spatial_buffer'] == True:
            buffer_mask_values = []
        for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
            # X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)            
            X, Y_depths = X.to(self.device), Y_depths.to(self.device)            

            # ======= Predict
            if isinstance(self.model, FocusOnDepth):
                _, output_segmentations = self.model(X)
            else:
                if getEncoder == True:
                    # print(len(self.model(X)))
                    # pdb.set_trace()
                    encoder_features, output_segmentations = self.model(X)
                    encoder_features = encoder_features.mean((2, 3))
                else:
                    _, output_segmentations = (None, self.model(X))

            
            softmax_segmentation = output_segmentations.cpu().detach().numpy()

            output = softmax_segmentation.argmax(axis=1).astype(np.uint8)

            output_values.append(output)
            reference_values.append(Y_segmentations.squeeze(1).detach().numpy())

            # ========= Apply softmax
            softmax_segmentation = softmax(softmax_segmentation, axis=1)[:, 1]

            # ========= Get uncertainty            
            ## print(softmax_segmentation.shape)
            pred_entropy_batch = []
            if self.config['ActiveLearning']['spatial_buffer'] == True:
                buffer_mask_batch = []
            for idx in range(len(softmax_segmentation)):
                pred_entropy = uncertainty.single_experiment_entropy(
                        np.expand_dims(softmax_segmentation[idx], axis=-1)).astype(np.float32)
                
                
                if self.config['ActiveLearning']['spatial_buffer'] == True:
                    pred_entropy, buffer_mask = uncertainty.apply_spatial_buffer(
                        pred_entropy, softmax_segmentation[idx]
                    )
                    buffer_mask_batch.append(np.expand_dims(buffer_mask, axis=0))
                
                pred_entropy_batch.append(np.expand_dims(pred_entropy, axis=0))
            pred_entropy_batch = np.concatenate(pred_entropy_batch, axis=0)
            # print("pred_entropy_batch.shape", pred_entropy_batch.shape)
            uncertainty_values.append(pred_entropy_batch)
            if self.config['ActiveLearning']['spatial_buffer'] == True:
                buffer_mask_batch = np.concatenate(buffer_mask_batch, axis=0)
                buffer_mask_values.append(buffer_mask_batch)

            if getEncoder == True:
                encoder_value = encoder_features.cpu().detach().numpy()
                encoder_values.append(encoder_value)
        output_values = np.concatenate(output_values, axis=0)
        uncertainty_values = np.concatenate(uncertainty_values, axis=0)
        reference_values = np.concatenate(reference_values, axis=0)
        if self.config['ActiveLearning']['spatial_buffer'] == True:
            buffer_mask_values = np.concatenate(buffer_mask_values, axis=0)
            self.activeLearner.setBufferMaskValues(buffer_mask_values)
            print("buffer_mask_values.shape", buffer_mask_values.shape)

        print(output_values.shape, uncertainty_values.shape, reference_values.shape) 
        if getEncoder == True:
            encoder_values = np.concatenate(encoder_values, axis=0)
            print("encoder_values.shape", encoder_values.shape)
            return output_values, reference_values, uncertainty_values, encoder_values
        return output_values, reference_values, uncertainty_values, _

class PredictorMCDropout(Predictor):
    def __init__(self, config, input_images):
        super().__init__(config, input_images)
        enable_dropout(self.model)
        self.inference_times = 10

        # check_dropout_enabled(self.model)
        # exit(0)

    def run(self):
        with torch.no_grad():
            for images in self.input_images:
                pil_im = Image.open(images)
                original_size = pil_im.size

                tensor_im = self.transform_image(pil_im).unsqueeze(0)

                softmax_segmentations = []

                for tm in range(self.inference_times):
                    # print('time: ', tm)
                    if isinstance(self.model, FocusOnDepth):
                        _, output_segmentation = self.model(tensor_im)
                    else:
                        _, output_segmentation = (None, self.model(tensor_im))
                    # output_depth = 1-output_depth
                    # output_depth  = output_depth.squeeze(0)
                    output_segmentation = output_segmentation.squeeze(0)

                    softmax_segmentation = output_segmentation.cpu().detach().numpy()
                    softmax_segmentation = softmax(softmax_segmentation, axis=0)[1] # get foreground class
                    
                    # ic(softmax_segmentation.shape)
                    softmax_segmentations.append(np.expand_dims(softmax_segmentation, axis=0))
                    # pdb.set_trace()
                softmax_segmentations = np.concatenate(softmax_segmentations, axis=0)
                softmax_segmentations = np.expand_dims(softmax_segmentations, axis=-1)
                ic(softmax_segmentations.shape)

                [pred_entropy, pred_var, MI, KL] = uncertainty.getUncertaintyMetrics(softmax_segmentations)

                ic(pred_entropy.shape)

                output_segmentation = transforms.ToPILImage()(output_segmentation.argmax(dim=0).float()).resize(original_size, resample=Image.NEAREST)
                # output_depth = transforms.ToPILImage()(output_depth.float()).resize(original_size, resample=Image.BICUBIC)

                ic(original_size)
                # pdb.set_trace()
                pred_entropy = cv2.resize(pred_entropy, original_size, interpolation=cv2.INTER_LINEAR)
                pred_var = cv2.resize(pred_var, original_size, interpolation=cv2.INTER_LINEAR)
                MI = cv2.resize(MI, original_size, interpolation=cv2.INTER_LINEAR)
                KL = cv2.resize(KL, original_size, interpolation=cv2.INTER_LINEAR)

                # pdb.set_trace()
                # exit(0)
                
                path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')
                path_dir_depths = os.path.join(self.output_dir, 'depths')

                path_dir_uncertainty_pred_entropy = os.path.join(self.output_dir, 'uncertainty_pred_entropy')
                path_dir_uncertainty_pred_var = os.path.join(self.output_dir, 'uncertainty_pred_var')
                path_dir_uncertainty_MI = os.path.join(self.output_dir, 'uncertainty_MI')
                path_dir_uncertainty_KL = os.path.join(self.output_dir, 'uncertainty_KL')

                create_dir(path_dir_segmentation)
                output_segmentation.save(os.path.join(path_dir_segmentation, os.path.basename(images)))

                # path_dir_depths = os.path.join(self.output_dir, 'depths')
                # create_dir(path_dir_depths)
                # output_depth.save(os.path.join(path_dir_depths, os.path.basename(images)))


                create_dir(path_dir_uncertainty_pred_entropy)
                plt.imshow(pred_entropy, cmap = plt.cm.gray)
                plt.axis('off')
                plt.savefig(os.path.join(path_dir_uncertainty_pred_entropy, os.path.basename(images)), 
                    dpi=150, bbox_inches='tight', pad_inches=0.0)

                create_dir(path_dir_uncertainty_pred_var)
                plt.imshow(pred_var, cmap = plt.cm.gray)
                plt.axis('off')
                plt.savefig(os.path.join(path_dir_uncertainty_pred_var, os.path.basename(images)), 
                    dpi=150, bbox_inches='tight', pad_inches=0.0)

                create_dir(path_dir_uncertainty_MI)
                plt.imshow(MI, cmap = plt.cm.gray)
                plt.axis('off')
                plt.savefig(os.path.join(path_dir_uncertainty_MI, os.path.basename(images)), 
                    dpi=150, bbox_inches='tight', pad_inches=0.0)

                create_dir(path_dir_uncertainty_KL)
                plt.imshow(KL, cmap = plt.cm.gray)
                plt.axis('off')
                plt.savefig(os.path.join(path_dir_uncertainty_KL, os.path.basename(images)), 
                    dpi=150, bbox_inches='tight', pad_inches=0.0)


                # cv2.imwrite(os.path.join(path_dir_uncertainty, os.path.basename(images)), pred_entropy)

class PredictorEntropy(Predictor):
    def __init__(self, config, input_images):
        super().__init__(config, input_images)

        # check_dropout_enabled(self.model)
        # exit(0)

    def run(self):
        with torch.no_grad():
            for images in self.input_images:
                pil_im = Image.open(images)
                original_size = pil_im.size

                tensor_im = self.transform_image(pil_im).unsqueeze(0)



                if isinstance(self.model, FocusOnDepth):
                    _, output_segmentation = self.model(tensor_im)
                else:
                    _, output_segmentation = (None, self.model(tensor_im))

                output_segmentation = output_segmentation.squeeze(0)

                softmax_segmentation = output_segmentation.cpu().detach().numpy()
                ic(softmax_segmentation.shape)
                softmax_segmentation = softmax(softmax_segmentation, axis=0)[1] # get foreground class
                ic(softmax_segmentation.shape)
                
                pred_entropy = uncertainty.single_experiment_entropy(
                    np.expand_dims(softmax_segmentation, axis=-1)).astype(np.float32)
                

                ic(pred_entropy.shape)

                output_segmentation = transforms.ToPILImage()(output_segmentation.argmax(dim=0).float()).resize(original_size, resample=Image.NEAREST)

                pred_entropy = cv2.resize(pred_entropy, original_size, interpolation=cv2.INTER_LINEAR)

                path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')

                path_dir_uncertainty_pred_entropy = os.path.join(self.output_dir, 'uncertainty_pred_entropy_single')

                create_dir(path_dir_segmentation)
                output_segmentation.save(os.path.join(path_dir_segmentation, os.path.basename(images)))

                create_dir(path_dir_uncertainty_pred_entropy)
                plt.imshow(pred_entropy, cmap = plt.cm.gray)
                plt.axis('off')
                plt.savefig(os.path.join(path_dir_uncertainty_pred_entropy, os.path.basename(images)), 
                    dpi=150, bbox_inches='tight', pad_inches=0.0)

from FOD.dataset import AutoFocusDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def createSaveFolders(output_dir):
    path_dir_segmentation = os.path.join(output_dir, 'segmentations')
    path_dir_reference = os.path.join(output_dir, 'reference')

    path_dir_uncertainty_pred_entropy = os.path.join(output_dir, 'uncertainty_pred_entropy_single')

    create_dir(path_dir_segmentation)
    create_dir(path_dir_reference)

    return path_dir_segmentation, path_dir_reference, path_dir_uncertainty_pred_entropy

def saveImages(reference_value, output_segmentation, pred_entropy, filename, 
        path_dir_segmentation, path_dir_reference, path_dir_uncertainty_pred_entropy):
    print(np.unique(output_segmentation))

    print(np.unique(reference_value))
    images = filename

    # output_segmentation = transforms.ToPILImage()(output_segmentation)# .resize(original_size, resample=Image.NEAREST)


    print(os.path.join(path_dir_segmentation), os.path.basename(images))
    cv2.imwrite(os.path.join(path_dir_segmentation, os.path.basename(images)), output_segmentation*255)
    cv2.imwrite(os.path.join(path_dir_reference, os.path.basename(images)), reference_value*255)

    create_dir(path_dir_uncertainty_pred_entropy)
    plt.imshow(pred_entropy, cmap = plt.cm.gray)
    plt.axis('off')
    plt.savefig(os.path.join(path_dir_uncertainty_pred_entropy, os.path.basename(images)), 
        dpi=150, bbox_inches='tight', pad_inches=0.0)
        
def getSortedFilename(filename):

    filename = filename.split("\\")[-1]
    filename_sorted = filename.split('.')
    filename_sorted = filename_sorted[0] + '_sorted.' + filename_sorted[1]
    return filename_sorted

def getFilename(path):

    filename = path.split("\\")[-1]
    return filename

from sklearn.utils import class_weight

class PredictorTrain(Predictor):
    def __init__(self, config, input_images):
        super().__init__(config, input_images)

        # check_dropout_enabled(self.model)
        # exit(0)

    def run(self):
        
        list_data = self.config['Dataset']['paths']['list_datasets']

        autofocus_datasets_test = []
        test_data = AutoFocusDataset(self.config, list_data[0], 'train')
        test_dataloader = DataLoader(test_data, batch_size=self.config['General']['test_batch_size'], shuffle=False)

        pbar = tqdm(test_dataloader)
        pbar.set_description("Testing")
        self.model.to(self.device)
        
        softmax_segmentations = []
        output_values = []
        uncertainty_values = []
        reference_values = []
        for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
            reference_values.append(Y_segmentations.squeeze(1).detach().numpy())
        reference_values = np.concatenate(reference_values, axis=0)
        class_weights = class_weight.compute_class_weight(
           'balanced',
            classes = np.unique(reference_values.flatten()), 
            y = reference_values.flatten())
        print(class_weights)


class PredictorEntropyAL(Predictor):
    def __init__(self, config, input_folder_path):
        super().__init__(config, input_folder_path)
        self.input_folder_path = input_folder_path
        self.activeLearner = ActiveLearner(config)


    def run(self):
        t0 = time.time()

        get_metrics = self.config['Inference']['get_metrics']

        config_active_learning = copy.deepcopy(self.config)
        config_active_learning['Dataset']['splits']['split_train'] = 0.
        config_active_learning['Dataset']['splits']['split_val'] = 0.
        config_active_learning['Dataset']['splits']['split_test'] = 1.

        test_data = AutoFocusDataset(config_active_learning, 
            self.input_folder_path, 'test')
        print(len(test_data.paths_images))

        # ================= Select random indexes if active learning method is random
        if self.config['ActiveLearning']['method'] == 'random':
            self.activeLearner.getRandomIdxs(test_data, 
                self.config['ActiveLearning']['k'])
            ic(self.config['ActiveLearning']['method'], len(self.activeLearner.recommendation_idxs))
            
            self.activeLearner.saveRecommendationIdxs()
            sys.exit(0)

        # ================= Predict test data
        test_dataloader = DataLoader(test_data, batch_size=self.config['General']['test_batch_size'], shuffle=False)
        if self.config['ActiveLearning']['diversity_method'] != False:
            output_values, reference_values, uncertainty_values, encoder_values = self.inferDataLoader(
                test_dataloader, getEncoder=True)
        else:
            output_values, reference_values, uncertainty_values, _ = self.inferDataLoader(
                test_dataloader)
        
        
        if get_metrics == True:
            f1 = metrics.f1_score(reference_values.flatten(), output_values.flatten())
            print("f1:", f1)
            oa = metrics.accuracy_score(reference_values.flatten(), output_values.flatten())
            print("oa:", oa)
        

        if self.config['ActiveLearning']['diversity_method'] == 'distance_to_train':
            train_data = AutoFocusDataset(self.config, 'CorrosaoTrainTest', 'train')
            train_dataloader = DataLoader(train_data, batch_size=self.config['General']['test_batch_size'], shuffle=False)

            _, _, _, train_encoder_values = self.inferDataLoader(
                train_dataloader, getEncoder=True)
            self.activeLearner.setTrainEncoderValues(train_encoder_values)


        self.activeLearner.getTopRecommendations(uncertainty_values, encoder_values)

        if self.config['ActiveLearning']['random_percentage'] > 0:
            self.activeLearner.getRandomIdxsForPercentage(test_data)

        self.activeLearner.saveRecommendationIdxs()

        print("recommendation IDs", self.activeLearner.recommendation_idxs)
        
        print("sorted mean uncertainty", self.activeLearner.sorted_values)

        if self.save_images == True:
            print("Starting to save images...")
            path_dir_segmentation, path_dir_reference, path_dir_uncertainty_pred_entropy = createSaveFolders(
                    self.output_dir)
            for idx in range(len(output_values)):

                filename = getFilename(test_data.paths_images[idx])
                
                
                saveImages(reference_values[idx], output_values[idx], 
                    uncertainty_values[idx], filename = filename, 
                    path_dir_segmentation = path_dir_segmentation,
                    path_dir_reference = path_dir_reference,
                    path_dir_uncertainty_pred_entropy = path_dir_uncertainty_pred_entropy
                    )





        print("time", time.time() - t0)
        if False:
            fig, axs = plt.subplots(2)
            axs[0].plot(np.array(oa_values)[recommendation_idxs])
            axs[0].set_ylabel('Overall Accuracy')
            axs[0].set_xlabel('Sample ID')
            axs[1].plot(sorted_values)
            axs[1].set_ylabel('Uncertainty')
            axs[1].set_xlabel('Sample ID')
            
            fig, axs = plt.subplots(2)
            axs[0].plot(np.array(f1_values)[recommendation_idxs])
            axs[0].set_ylabel('F1 Score')
            axs[0].set_xlabel('Sample ID')
            axs[1].plot(sorted_values)
            axs[1].set_ylabel('Uncertainty')
            axs[1].set_xlabel('Sample ID')

            plt.figure()
            plt.scatter(sorted_values, np.array(oa_values)[recommendation_idxs])
            plt.xlabel('Uncertainty')
            plt.ylabel('Overall Accuracy')
            plt.figure()
            plt.scatter(sorted_values, np.array(f1_values)[recommendation_idxs])
            plt.xlabel('Uncertainty')
            plt.ylabel('F1')
        
        # plt.show()
        # pdb.set_trace()

        # output_values, reference_values, uncertainty_values

        # ============= save images




class PredictorWithMetrics(Predictor):
    def __init__(self, config, input_images):
        super().__init__(config, input_images)

        # check_dropout_enabled(self.model)
        # exit(0)


    def run(self):
        
        list_data = self.config['Dataset']['paths']['list_datasets']

        test_data = AutoFocusDataset(self.config, list_data[0], 'test')
        print(len(test_data.paths_images))
        # pdb.set_trace()
        test_dataloader = DataLoader(test_data, batch_size=self.config['General']['test_batch_size'], shuffle=False)

        if self.config['ActiveLearning']['diversity_method'] != False:
            output_values, reference_values, uncertainty_values, encoder_values = self.inferDataLoader(
                test_dataloader, getEncoder=True)
        else:
            output_values, reference_values, uncertainty_values, _ = self.inferDataLoader(
                test_dataloader)
        
        # print(np.unique(output_values, return_counts=True))
        # print(np.unique(reference_values, return_counts=True))
        

        f1 = metrics.f1_score(reference_values.flatten(), output_values.flatten())
        print("f1:", f1)
        oa = metrics.accuracy_score(reference_values.flatten(), output_values.flatten())
        print("oa:", oa)

'''

    def run(self):
        t0 = time.time()
        labels = []
        output_segmentations = []
        with torch.no_grad():

            for images in self.input_images:
                pil_im = Image.open(images)
                original_size = pil_im.size

                tensor_im = self.transform_image(pil_im).unsqueeze(0)
                if isinstance(self.model, FocusOnDepth):
                    _, output_segmentation = self.model(tensor_im)
                else:
                    _, output_segmentation = (None, self.model(tensor_im))
                
                output_segmentation = transforms.ToPILImage()(output_segmentation.squeeze(0).argmax(dim=0).float()).resize(original_size, resample=Image.NEAREST)

                output_segmentation = np.array(output_segmentation)
                output_segmentation[output_segmentation>0] = 1
                # print(output_segmentation.shape)
                
                # print(images)
                label_path = images.split("\\")

                label_path = label_path[0] + '/' + label_path[1] + '/' + 'labels/' + label_path[-1]
                # print(label_path)
                label = Image.open(label_path)
                label = np.array(label)
                label = label[...,0]
                label[label>0] = 1
                # print(label.shape)
                # print(np.unique(label))
                # print(label.dtype, output_segmentation.dtype)
                try:
                    labels.append(np.expand_dims(label.flatten(), axis=0))
                    output_segmentations.append(np.expand_dims(output_segmentation.flatten(), axis=0))
                except:
                    break
        labels = np.concatenate(labels, axis=None).flatten().squeeze().astype(np.uint8)
        output_segmentations = np.concatenate(output_segmentations, axis=None).flatten().squeeze().astype(np.uint8)


        print(labels.shape, output_segmentations.shape)
        print("Time", time.time() - t0)

        miou = metrics.jaccard_score(labels, output_segmentations)
        print("mIoU:", miou)
        print(metrics.classification_report(labels, output_segmentations))

        f1 = metrics.f1_score(labels, output_segmentations)
        print("f1:", f1)

        print("Time", time.time() - t0)

        exit(0)
'''