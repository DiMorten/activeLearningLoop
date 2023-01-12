import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pdb
from scipy.spatial import distance
from icecream import ic
import pandas as pd
import shutil
import pathlib
from glob import glob
from src.CubemapHandler import CubemapHandler
from natsort import natsorted
import os
import sys
import utils

class ActiveLearner():
    def __init__(self, config):
        self.config = config
        self.cubemapHandler = CubemapHandler(config['cubemap_keyword'])

        self.k = self.config['k']
        self.recommendation_idxs_path = self.config['output_path'] + \
            '/inference/recommendation_idxs_' + \
            str(self.config['exp_id']) + '.npy'

    def setTrainEncoderValues(self, train_encoder_values):
        self.train_encoder_values = train_encoder_values

    def setBufferMaskValues(self, buffer_mask_values):
        self.buffer_mask_values = buffer_mask_values

    def getTopRecommendations(self, uncertainty_values_mean, encoder_values, train_encoder_values = None):

        
        if self.config['diversity_method'] == "None" or self.config['diversity_method'] == None:        
            K = self.k
        else:
            K = self.k * self.config['beta']

        print("self.k, K", self.k, K)
        # pdb.set_trace()
        # K = 20
        # self.k = 10
        # uncertainty_values_mean = self.getUncertaintyMean(uncertainty_values)

        self.sorted_values, self.recommendation_idxs = utils.getTopRecommendations(
            uncertainty_values_mean, K=K)

        # ic(self.recommendation_idxs)
        # pdb.set_trace()
        ## print("sorted_values.shape", self.sorted_values.shape)
        
        # print("sorted name IDs", np.array([x.split("\\")[-1] for x in paths_images])[self.recommendation_idxs])


        ## print("sorted mean uncertainty number", self.sorted_values)

        if self.config['diversity_method'] == 'cluster':   
            representative_idxs, self.recommendation_idxs = utils.getRepresentativeSamplesFromCluster(
                encoder_values[self.recommendation_idxs], 
                self.recommendation_idxs, 
                k=self.k)
                # n_components = self.config['cluster_n_components'])

            self.sorted_values = self.sorted_values[representative_idxs]
        elif self.config['diversity_method'] == 'distance_to_train':

            representative_idxs, self.recommendation_idxs = utils.getRepresentativeSamplesFromDistance(
                encoder_values[self.recommendation_idxs], 
                self.recommendation_idxs, 
                train_values = train_encoder_values, 
                k=self.k)
                
            self.sorted_values = self.sorted_values[representative_idxs]
        
    '''
    def getRandomIdxs(self, dataset, n):
        self.recommendation_idxs = getRandomIdxs(dataset, n)
        
        # return idxs
    '''
    def getRandomIdxsForPercentage(self, len_vector):
        sample_n_with_random_percentage = int(
            self.config['k'] * self.config['random_percentage'])

        print("sample_n with random percentage:", sample_n_with_random_percentage)

        recommendation_idxs_with_random_percentage = utils.getRandomIdxs(len_vector, 
            sample_n_with_random_percentage)

        self.recommendation_idxs[-sample_n_with_random_percentage:] = recommendation_idxs_with_random_percentage

    def saveRecommendationIdxs(self):

        np.save(self.recommendation_idxs_path, 
            self.recommendation_idxs)

    def getSelectedImageNames(self, paths_images):
        self.query_image_names = np.array(
            [os.path.basename(x) for x in paths_images])[self.recommendation_idxs]

    def saveSelectedImageNames(self, query_image_names, ext = "png"):
        
        query_image_names = ["{}.{}".format(x.split('.')[0], ext) for x in query_image_names]
        print(
            "sorted name IDs", query_image_names)

        #  convert array into dataframe
        df = pd.DataFrame(query_image_names)
        df = df.reset_index(drop=True)
        # save the dataframe as a csv file
        path = pathlib.Path(
            self.config['output_path'] + \
                '/active_learning/')
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(path / "query_image_names.csv"),
            index=False, header=False)


    def getFilenamesFromPaths(self, paths):
        # return [x.split('\\')[-1] for x in paths]
        return [os.path.basename(x) for x in paths]

    def loadData(self):

        self.inferenceResults = lambda: None
        self.inferenceResults.paths_images = utils.getFilenamesFromFolderList(
            [os.path.join(
            self.config['output_path'], 'uncertainty_map', x) for x in self.config['output_folders']])

        self.inferenceResults.paths_images = [os.path.basename(x) for x in self.inferenceResults.paths_images]
        self.inferenceResults.uncertainty_values = utils.loadFromFolder(
            [os.path.join(self.config['output_path'], 'uncertainty_map', x) for x in self.config['output_folders']]
        )
        self.inferenceResults.encoder_values = utils.loadFromFolder(
            [os.path.join(self.config['output_path'], 'aspp_features', x) for x in self.config['output_folders']]
        )
        if self.config['diversity_method'] == 'distance_to_train':
            self.inferenceResults.train_encoder_values = utils.loadFromFolder(
                [os.path.join(self.config['output_path'].split('/')[0]+'_train', 'aspp_features') for x in self.config['output_folders']]
            )

        print(self.inferenceResults.encoder_values.shape)
        # pdb.set_trace()
        print(self.inferenceResults.uncertainty_values.shape)
        # pdb.set_trace()
        
    def run(self):

        # pdb.set_trace()
        filenames = self.getFilenamesFromPaths(
            self.inferenceResults.paths_images)

        self.cubemapHandler.findCubemapFiles(filenames)
        
        # =========== Get image mean for uncertainty values

        if self.config['spatial_buffer'] == False:
            self.inferenceResults.uncertainty_values_mean = utils.getUncertaintyMean(
                self.inferenceResults.uncertainty_values)
        else:
            self.inferenceResults.uncertainty_values_mean = utils.getUncertaintyMeanBuffer(
                self.inferenceResults.uncertainty_values, self.buffer_mask_values)

        # =========== Treat cubemap samples as single 360 image

        self.inferenceResults.uncertainty_values_mean, _ = self.cubemapHandler.reduceArray(
            self.inferenceResults.uncertainty_values_mean)
        self.inferenceResults.uncertainty_values, _ = self.cubemapHandler.reduceArray(
            self.inferenceResults.uncertainty_values)
        self.inferenceResults.encoder_values, _ = self.cubemapHandler.reduceArray(
            self.inferenceResults.encoder_values)
        self.inferenceResults.paths_images_not_reduced = self.inferenceResults.paths_images.copy()
        self.inferenceResults.paths_images = list(
            self.cubemapHandler.reduceFilenames(
                np.array(self.inferenceResults.paths_images)))

        # self.len_vector = len(self.inferenceResults.paths_images)
        
        self.len_vector = self.inferenceResults.uncertainty_values_mean.shape[0]


        # self.cubemapHandler.reduceArray(
        #     self.inferenceResults.uncertainty_values)

        # if self.config['diversity_method'] == 'distance_to_train':
        #     self.setTrainEncoderValues(self.train_encoder_values)

        
        if self.config['random_percentage'] == 1:
            # If 100% random selection, select and exit early
            self.recommendation_idxs = utils.getRandomIdxs(self.len_vector, len(self.len_vector))
        else:
            if self.config['diversity_method'] == "cluster":
                self.getTopRecommendations(self.inferenceResults.uncertainty_values_mean, 
                    self.inferenceResults.encoder_values)
            elif self.config['diversity_method'] == "distance_to_train":
                self.getTopRecommendations(self.inferenceResults.uncertainty_values_mean, 
                    self.inferenceResults.encoder_values, 
                    train_encoder_values = self.inferenceResults.train_encoder_values)
            else:
                self.getTopRecommendations(self.inferenceResults.uncertainty_values_mean, None)

            if self.config['random_percentage'] > 0:
                self.getRandomIdxsForPercentage(self.len_vector)
       
        self.saveRecommendationIdxs()

        # self.query_image_names = [os.path.basename(x) for x in self.inferenceResults.paths_images]
        self.getSelectedImageNames(self.inferenceResults.paths_images)
        # pdb.set_trace()

        # self.saveSelectedImages(self.query_image_names)

        # print("self.query_image_names:", self.query_image_names)
        query_image_names_not_reduced = self.cubemapHandler.getCubemapFilenamesFromSingleFaceNames(
            self.inferenceResults.paths_images_not_reduced,
            self.query_image_names
        )
        # ic(len(self.query_image_names))
        
        # ic(len(query_image_names_not_reduced))

        self.saveSelectedImageNames(query_image_names_not_reduced)

        self.saveSelectedImages(query_image_names_not_reduced)



        print("recommendation IDs", self.recommendation_idxs)
        
        print("sorted mean uncertainty", self.sorted_values)
    
    def copy_files_to_folder(self, input_path, output_path,
        query_image_names, ext = 'png'):
        save_path = pathlib.Path(
            output_path)
        save_path.mkdir(parents=True, exist_ok=True)
    
        for file in query_image_names:
            try:
                file = '{}.{}'.format(file.split('.')[0], ext)
                shutil.copyfile(input_path + file, 
                    str(save_path / file))
            except Exception as e:
                print(e)
    def saveSelectedImages(self, query_image_names):
        
        print(query_image_names)

        self.copy_files_to_folder(
            input_path = self.config['image_path'],
            output_path = self.config['output_path'] + '/active_learning/query_images/imgs/',
            query_image_names = query_image_names,
            ext = 'png'
            )

        self.copy_files_to_folder(
            input_path = self.config['output_path'] + \
                    '/segmentations/',
            output_path = self.config['output_path'] + '/active_learning/query_images/segmentations/',
            query_image_names = query_image_names,
            ext = 'png'            
            )

        self.copy_files_to_folder(
            input_path = self.config['output_path'] + \
                    '/uncertainty_map/',
            output_path = self.config['output_path'] + '/active_learning/query_images/uncertainty/',
            query_image_names = query_image_names,
            ext = 'npz'
            )

