import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pdb
from scipy.spatial import distance
from icecream import ic

def getTopRecommendations(values, K=500, mode='uncertainty'):
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
