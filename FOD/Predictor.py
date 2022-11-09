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
from FOD.ResUnet import ResUnetPlusPlus

from FOD.utils import create_dir
from FOD.dataset import show
import FOD.uncertainty as uncertainty
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

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        # resize = 513
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

        # path_model = os.path.join(config['General']['path_model'], 'FocusOnDepth_{}.p'.format(config['General']['model_timm']))
        path_model = os.path.join(config['General']['path_model'], 'ResUnetPlusPlus.p')
        
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.eval()

        self.transform_image = transforms.Compose([
            transforms.Resize((resize, resize)),
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
                _, output_segmentation = self.model(tensor_im)
                # output_depth = 1-output_depth

                output_segmentation = transforms.ToPILImage()(output_segmentation.squeeze(0).argmax(dim=0).float()).resize(original_size, resample=Image.NEAREST)
                # output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)

                path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')
                path_dir_depths = os.path.join(self.output_dir, 'depths')
                create_dir(path_dir_segmentation)
                output_segmentation.save(os.path.join(path_dir_segmentation, os.path.basename(images)))

                # path_dir_depths = os.path.join(self.output_dir, 'depths')
                # create_dir(path_dir_depths)
                # output_depth.save(os.path.join(path_dir_depths, os.path.basename(images)))

                ## TO DO: Apply AutoFocus

                # output_depth = np.array(output_depth)
                # output_segmentation = np.array(output_segmentation)

                # mask_person = (output_segmentation != 0)
                # depth_person = output_depth*mask_person
                # mean_depth_person = np.mean(depth_person[depth_person != 0])
                # std_depth_person = np.std(depth_person[depth_person != 0])

                # #print(mean_depth_person, std_depth_person)

                # mask_total = (depth_person >= mean_depth_person-2*std_depth_person)
                # mask_total = np.repeat(mask_total[:, :, np.newaxis], 3, axis=-1)
                # region_to_blur = np.ones(np_im.shape)*(1-mask_total)

                # #region_not_to_blur = np.zeros(np_im.shape) + np_im*(mask_total)
                # region_not_to_blur = np_im
                # blurred = cv2.blur(region_to_blur, (10, 10))

                # #final_image = blurred + region_not_to_blur
                # final_image = cv2.addWeighted(region_not_to_blur.astype(np.uint8), 0.5, blurred.astype(np.uint8), 0.5, 0)
                # final_image = Image.fromarray((final_image).astype(np.uint8))
                # final_image.save(os.path.join(self.output_dir, os.path.basename(images)))

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

                    output_depth, output_segmentation = self.model(tensor_im)
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




class PredictorSingleEntropy(Predictor):
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

                softmax_segmentations = []


                output_depth, output_segmentation = self.model(tensor_im)


                # output_depth = 1-output_depth
                # output_depth  = output_depth.squeeze(0)
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

                # pdb.set_trace()
                # exit(0)
                
                path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')

                path_dir_uncertainty_pred_entropy = os.path.join(self.output_dir, 'uncertainty_pred_entropy_single')

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

