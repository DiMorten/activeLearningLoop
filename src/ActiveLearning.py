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
def getFilenamesFromFolder(path):
    filenames = []
    for name in glob(path + '/*.npz'):
        filenames.append(name)
    return filenames

def getFilenamesFromFolderList(paths):
    filenames = []
    for path in paths:
        filenames.extend(getFilenamesFromFolder(path))
    return filenames
def loadFromFolder(paths):
    filenames = getFilenamesFromFolderList([os.path.join(path) for path in paths])
    for idx, filename in enumerate(filenames):
        if idx == 0:
            values = np.expand_dims(np.load(filename)['arr_0'], axis=0)
        else:
            values = np.concatenate((values, 
                np.expand_dims(np.load(filename)['arr_0'], axis=0)), axis=0)
    return values

def getUncertaintyMean(values):
    mean_values = np.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)
    return mean_values

def getUncertaintyMeanBuffer(values, buffer_mask_values):
    print(buffer_mask_values.shape)
    # pdb.set_trace()
    values = np.ma.array(values, mask = buffer_mask_values)
    mean_values = np.ma.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)

def getTopRecommendations(mean_values, K=500):
    # values: shape (N, h, w)

    sorted_idxs = np.argsort(mean_values, axis=0)
    sorted_values = np.sort(mean_values, axis=0)

    recommendation_idxs = np.flip(sorted_idxs)[:K]
    return np.flip(sorted_values)[:K], recommendation_idxs


def getRepresentativeSamplesFromCluster(values, recommendation_idxs, k=250, 
    n_components = 100, mode='cluster'):

    '''
    values: shape (n_samples, feature_len)
    '''

    pca = PCA(n_components = n_components)
    pca.fit(values)
    values = pca.transform(values)
    # print(pca.explained_variance_ratio_)
    # pdb.set_trace()

    cluster = KMeans(n_clusters = k)
    print("Fitting cluster...")
    distances_to_centers = cluster.fit_transform(values)

    print("...Finished fitting cluster.")
    


    print("values.shape", values.shape)
    print("distances_to_centers.shape", distances_to_centers.shape)

    representative_idxs = []
    for k_idx in range(k):
       representative_idxs.append(np.argmin(distances_to_centers[:, k_idx]))
    representative_idxs = np.array(representative_idxs)
    representative_idxs = np.sort(representative_idxs, axis=0)
    print(representative_idxs)
    # pdb.set_trace()

    # print("getRepresentativeSamples")
    

    # return representative_idxs, # recommendation_idxs[selected_samples]
    return representative_idxs, recommendation_idxs[representative_idxs]

    # values: shape (N, h, w)


def getDistanceToList(value, train_values):
    distance_to_train = np.inf
    for train_value in train_values:
        distance_ = distance.cosine(value, train_value)
        if distance_ < distance_to_train:
            distance_to_train = distance_
    return distance_to_train
def getSampleWithLargestDistance(distances, mask):
    distances = np.ma.array(distances, mask = mask)
    # ic(np.ma.count_masked(distances))
    # ic(np.unique(mask, return_counts=True))
    # pdb.set_trace()
    return np.ma.max(distances, fill_value=0), np.ma.argmax(distances, fill_value=0)

def getRepresentativeSamplesFromDistance(values, recommendation_idxs, train_values, k=250, mode='max_k_cover'):

    '''
    values: shape (n_samples, feature_len)
    train_values: shape (train_n_samples, feature_len)
    '''

    pca = PCA(n_components = 100)
    pca.fit(values)
    values = pca.transform(values)

    train_values = pca.transform(train_values)
    
    distances_to_train = []
    representative_idxs = []
    for value in values:
        distance_to_train = getDistanceToList(value, train_values)
        distances_to_train.append(distance_to_train)
    distances_to_train = np.array(distances_to_train)

    values_selected_mask = np.zeros((len(values)), dtype=np.bool)
    for k_idx in range(k):
        print(k_idx)
        selected_sample, selected_sample_idx = getSampleWithLargestDistance(
            distances_to_train, 
            mask = values_selected_mask)
        representative_idxs.append(selected_sample_idx)
        
        values_selected_mask[selected_sample_idx] = True
        # values.pop(selected_sample_idx)
        # distances_to_train.pop(selected_sample_idx)
        '''
        distances_to_previously_selected_sample = getDistancesToValue(
            values,
            selected_sample_idx
        )
        '''
        for idx, value in enumerate(values):
            # ic(distances_to_train[idx])
            # ic(selected_sample)
            # pdb.set_trace()
            distances_to_train[idx] = np.minimum(distances_to_train[idx], 
                selected_sample)
    representative_idxs = np.array(representative_idxs)
    # print("1", values_selected_mask.argwhere(values_selected_mask == True))
    print("2", len(representative_idxs))
    representative_idxs = np.sort(representative_idxs, axis=0)
    
    return representative_idxs, recommendation_idxs[representative_idxs]

def getRepresentativeAndUncertain(values, recommendation_idxs, representative_idxs):
    return values[recommendation_idxs][representative_idxs] # enters N=100, returns k=10

def getRandomIdxs(len_vector, n):
    idxs = np.arange(len_vector)
    np.random.shuffle(idxs)
    idxs = idxs[:n]
    
    return idxs


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

        self.sorted_values, self.recommendation_idxs = getTopRecommendations(
            uncertainty_values_mean, K=K)

        # ic(self.recommendation_idxs)
        # pdb.set_trace()
        print("sorted_values.shape", self.sorted_values.shape)
        
        # print("sorted name IDs", np.array([x.split("\\")[-1] for x in paths_images])[self.recommendation_idxs])


        print("sorted mean uncertainty", self.sorted_values)
        if self.config['diversity_method'] == 'cluster':   
            representative_idxs, self.recommendation_idxs = getRepresentativeSamplesFromCluster(
                encoder_values[self.recommendation_idxs], 
                self.recommendation_idxs, 
                k=self.k,
                n_components = self.config['cluster_n_components'])

            self.sorted_values = self.sorted_values[representative_idxs]
        elif self.config['diversity_method'] == 'distance_to_train':

            representative_idxs, self.recommendation_idxs = getRepresentativeSamplesFromDistance(
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

        recommendation_idxs_with_random_percentage = getRandomIdxs(len_vector, 
            sample_n_with_random_percentage)

        self.recommendation_idxs[-sample_n_with_random_percentage:] = recommendation_idxs_with_random_percentage

    def saveRecommendationIdxs(self):

        np.save(self.recommendation_idxs_path, 
            self.recommendation_idxs)

    def getSelectedImageNames(self, paths_images):
        self.query_image_names = np.array(
            [os.path.basename(x) for x in paths_images])[self.recommendation_idxs]

    def saveSelectedImageNames(self, query_image_names):
        
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
        self.inferenceResults.paths_images = getFilenamesFromFolderList(
            [os.path.join(
            self.config['output_path'], 'uncertainty_map', x) for x in self.config['output_folders']])

        self.inferenceResults.paths_images = [os.path.basename(x) for x in self.inferenceResults.paths_images]
        self.inferenceResults.uncertainty_values = loadFromFolder(
            [os.path.join(self.config['output_path'], 'uncertainty_map', x) for x in self.config['output_folders']]
        )
        self.inferenceResults.encoder_values = loadFromFolder(
            [os.path.join(self.config['output_path'], 'encoder_features', x) for x in self.config['output_folders']]
        )

        print(self.inferenceResults.uncertainty_values.shape)
        # pdb.set_trace()
        
    def run(self):

        # pdb.set_trace()
        filenames = self.getFilenamesFromPaths(
            self.inferenceResults.paths_images)

        self.cubemapHandler.findCubemapFiles(filenames)
        
        # =========== Get image mean for uncertainty values

        if self.config['spatial_buffer'] == False:
            self.inferenceResults.uncertainty_values_mean = getUncertaintyMean(
                self.inferenceResults.uncertainty_values)
        else:
            self.inferenceResults.uncertainty_values_mean = getUncertaintyMeanBuffer(
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

        if self.config['diversity_method'] == 'distance_to_train':
            self.setTrainEncoderValues(self.train_encoder_values)


        if self.config['diversity_method'] != None and self.config['diversity_method'] != "None":
            self.getTopRecommendations(self.inferenceResults.uncertainty_values_mean, self.inferenceResults.encoder_values)
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
        query_image_names):
        save_path = pathlib.Path(
            output_path)
        save_path.mkdir(parents=True, exist_ok=True)
    
        for file in query_image_names:
            try:
                # file = '{}.png'.format(file.split('.')[0])
                shutil.copyfile(input_path + file, 
                    str(save_path / file))
            except Exception as e:
                print(e)
    def saveSelectedImages(self, query_image_names):
        
        print(query_image_names)

        self.copy_files_to_folder(
            input_path = self.config['image_path'],
            output_path = self.config['output_path'] + '/active_learning/query_images/imgs/',
            query_image_names = query_image_names
            )

        self.copy_files_to_folder(
            input_path = self.config['output_path'] + \
                    '/segmentations/',
            output_path = self.config['output_path'] + '/active_learning/query_images/segmentations/',
            query_image_names = query_image_names            
            )

        self.copy_files_to_folder(
            input_path = self.config['output_path'] + \
                    '/uncertainty_map/',
            output_path = self.config['output_path'] + '/active_learning/query_images/uncertainty/',
            query_image_names = query_image_names
            )

