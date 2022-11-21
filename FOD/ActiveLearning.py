import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
import pdb

def getTopRecommendations(values, K=500, mode='uncertainty'):
    # values: shape (N, h, w)

    mean_values = np.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)
    
    sorted_idxs = np.argsort(mean_values, axis=0)
    sorted_values = np.sort(mean_values, axis=0)

    recommendation_idxs = np.flip(sorted_idxs)[:K]
    return np.flip(sorted_values)[:K], recommendation_idxs

def getRepresentativeSamples(values, recommendation_idxs, k=250, mode='max_k_cover'):
    
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
    pdb.set_trace()

    print("getRepresentativeSamples")
    
    # return representative_idxs, # recommendation_idxs[selected_samples]
    return representative_idxs, recommendation_idxs[representative_idxs]

    # values: shape (N, h, w)

def getRepresentativeAndUncertain(values, recommendation_idxs, representative_idxs):
    return values[recommendation_idxs][representative_idxs] # enters N=100, returns k=10

