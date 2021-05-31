#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:44:00 2021

@author: Alaisha Naidu
Name: Intrinsic Dimension
Creds: DataCamp
Dataset: Iris Flower Dataset

"""
#Intrinsic Dimension of a dataset is the number of features needed to approximate the dataset
#Used for dimension reduction
#Intrinsic Dimension is found by counting the number of PCA Features that have a high variance

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iris = load_iris()
data = iris.data
species = iris.target #list of species of each sample

#create an array of just versicolor iris flower samples 
samples = []

for i in range(0, len(species)):
    if species[i]== 2:
        j = i
        samples.append(data[j])

#print(samples)

pca = PCA()
pca.fit(samples)
features = range(pca.n_components_)

#create a bar plot of the variances
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.title('Principal Components'' Variances')
plt.ylabel('Variance')
plt.xlabel('PCA Feature')
plt.show()


