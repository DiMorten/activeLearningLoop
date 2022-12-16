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

def getTopRecommendations(values, K=500):
    # values: shape (N, h, w)

    
    mean_values = np.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)
    
    sorted_idxs = np.argsort(mean_values, axis=0)
    sorted_values = np.sort(mean_values, axis=0)

    recommendation_idxs = np.flip(sorted_idxs)[:K]
    return np.flip(sorted_values)[:K], recommendation_idxs


def getTopRecommendationsBuffer(values, buffer_mask_values, K=500, mode='uncertainty'):
    # values: shape (N, h, w)

    print(buffer_mask_values.shape)
    # pdb.set_trace()
    values = np.ma.array(values, mask = buffer_mask_values)
    mean_values = np.ma.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)
    
    sorted_idxs = np.argsort(mean_values, axis=0)
    sorted_values = np.sort(mean_values, axis=0)

    recommendation_idxs = np.flip(sorted_idxs)[:K]
    return np.flip(sorted_values)[:K], recommendation_idxs


def getRepresentativeSamplesFromCluster(values, recommendation_idxs, k=250, mode='cluster'):

    '''
    values: shape (n_samples, feature_len)
    '''

    pca = PCA(n_components = 100)
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

def getRandomIdxs(dataset, n):
    idxs = np.arange(len(dataset.paths_images))
    np.random.shuffle(idxs)
    idxs = idxs[:n]
    
    return idxs


class ActiveLearner():
    def __init__(self, config, input_folder_path):
        self.config = config
        self.input_folder_path = input_folder_path
        ext = config['Dataset']['extensions']['ext_images']
        self.input_images = glob(input_folder_path + '/imgs/*' + ext)

        self.k = self.config['ActiveLearning']['k']
        self.recommendation_idxs_path = self.config['General']['path_predicted_images'] + \
            '/inference/recommendation_idxs_' + \
            str(self.config['General']['exp_id']) + '.npy'

    def setTrainEncoderValues(self, train_encoder_values):
        self.train_encoder_values = train_encoder_values

    def setBufferMaskValues(self, buffer_mask_values):
        self.buffer_mask_values = buffer_mask_values

    def getTopRecommendations(self, uncertainty_values, encoder_values, train_encoder_values = None):

        
        if self.config['ActiveLearning']['diversity_method'] == "None":        
            K = self.k
        else:
            K = self.k * self.config['ActiveLearning']['beta']

        print("self.k, K", self.k, K)
        pdb.set_trace()
        # K = 20
        # self.k = 10
        if self.config['ActiveLearning']['spatial_buffer'] == False:
            self.sorted_values, self.recommendation_idxs = getTopRecommendations(uncertainty_values, K=K)
        else:
            self.sorted_values, self.recommendation_idxs = getTopRecommendationsBuffer(
                uncertainty_values, self.buffer_mask_values, K=K)

        print("sorted_values.shape", self.sorted_values.shape)
        
        # print("sorted name IDs", np.array([x.split("\\")[-1] for x in test_data.paths_images])[self.recommendation_idxs])


        print("sorted mean uncertainty", self.sorted_values)
        if self.config['ActiveLearning']['diversity_method'] == 'cluster':   
            representative_idxs, self.recommendation_idxs = getRepresentativeSamplesFromCluster(
                encoder_values[self.recommendation_idxs], 
                self.recommendation_idxs, 
                k=self.k)

            self.sorted_values = self.sorted_values[representative_idxs]
        elif self.config['ActiveLearning']['diversity_method'] == 'distance_to_train':

            representative_idxs, self.recommendation_idxs = getRepresentativeSamplesFromDistance(
                encoder_values[self.recommendation_idxs], 
                self.recommendation_idxs, 
                train_values = train_encoder_values, 
                k=self.k)
                
            self.sorted_values = self.sorted_values[representative_idxs]
        

    def getRandomIdxs(self, dataset, n):
        self.recommendation_idxs = getRandomIdxs(dataset, n)
        
        # return idxs

    def getRandomIdxsForPercentage(self, dataset):
        sample_n_with_random_percentage = int(
            self.config['ActiveLearning']['k'] * self.config['ActiveLearning']['random_percentage'])

        print("sample_n_with_random_percentage", sample_n_with_random_percentage)
        recommendation_idxs_with_random_percentage = getRandomIdxs(dataset, 
            sample_n_with_random_percentage)

        self.recommendation_idxs[-sample_n_with_random_percentage:] = recommendation_idxs_with_random_percentage

    def saveRecommendationIdxs(self):

        np.save(self.recommendation_idxs_path, 
            self.recommendation_idxs)

    def getSelectedImageNames(self, dataset):
        self.query_image_names = np.array(
            [x.split("\\")[-1] for x in dataset.paths_images])[self.recommendation_idxs]

    def saveSelectedImageNames(self):
        
        print(
            "sorted name IDs", self.query_image_names)
        #  convert array into dataframe
        df = pd.DataFrame(self.query_image_names)
        df = df.reset_index(drop=True)
        # save the dataframe as a csv file
        path = pathlib.Path(
            self.config['General']['path_predicted_images'] + \
                '/active_learning/')
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(path / "query_image_names.csv"))



    def run(self, predictor):

        if self.config['ActiveLearning']['method'] == 'random':
            self.getRandomIdxs(predictor.test_data, 
                self.config['ActiveLearning']['k'])
            ic(self.config['ActiveLearning']['method'], len(self.recommendation_idxs))
            
            # self.saveRecommendationIdxs()
            # sys.exit(0)


        if self.config['ActiveLearning']['diversity_method'] == 'distance_to_train':
            self.setTrainEncoderValues(self.train_encoder_values)

        if self.config['ActiveLearning']['method'] != 'random':
            if self.config['ActiveLearning']['diversity_method'] != "None":
                self.getTopRecommendations(predictor.uncertainty_values, predictor.encoder_values)
            else:
                self.getTopRecommendations(predictor.uncertainty_values, None)


        if self.config['ActiveLearning']['random_percentage'] > 0:
            self.getRandomIdxsForPercentage(predictor.test_data)

        self.saveRecommendationIdxs()

        self.getSelectedImageNames(predictor.test_data)
        self.saveSelectedImageNames()
        self.saveSelectedImages()

        print("recommendation IDs", self.recommendation_idxs)
        
        print("sorted mean uncertainty", self.sorted_values)
    
    def saveSelectedImages(self):
        print(self.query_image_names)
        save_path = pathlib.Path(
            self.config['General']['path_predicted_images'] + '/active_learning/query_images/imgs/')
        save_path.mkdir(parents=True, exist_ok=True)

        for file in self.query_image_names:
            shutil.copyfile(self.input_folder_path + '/' + \
                self.config['Dataset']['paths']['path_images'] + '/' + file, 
                str(save_path / file))


        save_path = pathlib.Path(
            self.config['General']['path_predicted_images'] + '/active_learning/query_images/segmentations/')
        save_path.mkdir(parents=True, exist_ok=True)

        for file in self.query_image_names:
            # print("from", )
            shutil.copyfile(
                self.config['General']['path_predicted_images'] + \
                    '/segmentations/' + file, 
                str(save_path / file))


        save_path = pathlib.Path(
            self.config['General']['path_predicted_images'] + '/active_learning/query_images/uncertainty/')
        save_path.mkdir(parents=True, exist_ok=True)

        for file in self.query_image_names:

            shutil.copyfile(
                self.config['General']['path_predicted_images'] + \
                    '/uncertainty/' + file, 
                str(save_path / file))