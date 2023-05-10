
from sklearn.cluster  import KMeans
import numpy as np


def km(all_pixels,K):
    all_pixels=all_pixels.reshape(-1,1)
    km=KMeans(n_clusters=K)
    b=all_pixels.shape
    km.fit(all_pixels)
    centres=km.cluster_centers_
    centres=centres.reshape(1,5)
    a=km.labels_
    k=np.unique(a,return_counts=True)
    weights=np.divide(k[1],np.full((1,K),b[0]))

    return list(centres[0]),list(weights[0])




