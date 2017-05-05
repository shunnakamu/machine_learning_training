#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# load data_set
iris = datasets.load_iris()

# reduce dimension by PCA
iris_pca = PCA(n_components=2).fit(iris.data).transform(iris.data)

# clustering for all data
k_means_list = []
for i in range(3):
    k_means_list.append(KMeans(n_clusters=3, random_state=i).fit(iris.data))

# plot real class
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 16))
axes[0, 0].set_title("Real class", fontsize=40)
axes[0, 0].scatter(iris_pca[:, 0], iris_pca[:, 1], c=iris.target, s=80)

# plot clustered class
for i, ax in enumerate([axes[0, 1], axes[1, 0], axes[1, 1]]):
    ax.set_title("K-means cluster (%d)" % i, fontsize=40)
    ax.scatter(iris_pca[:, 0], iris_pca[:, 1], c=k_means_list[i].labels_, s=80)

fig.show()

# k-means for 5 clusters
k_means = KMeans(n_clusters=5).fit(iris.data)
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(20, 8))
axL.set_title("Real class", fontsize=40)
axL.scatter(iris_pca[:, 0], iris_pca[:, 1], c=iris.target, s=80)

axR.set_title("K-means cluster", fontsize=40)
axR.scatter(iris_pca[:, 0], iris_pca[:, 1], c=k_means.labels_, s=80)
fig.show()

