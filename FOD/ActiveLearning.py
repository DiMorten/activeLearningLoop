import numpy as np
def getTopRecommendations(values, k=5, mode='uncertainty'):
    # values: shape (N, h, w)

    mean_values = np.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)
    
    sorted_idxs = np.argsort(mean_values, axis=0)
    sorted_values = np.sort(mean_values, axis=0)
    return np.flip(sorted_values), np.flip(sorted_idxs)