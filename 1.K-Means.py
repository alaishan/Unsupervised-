#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:17:25 2021

@author: Alaisha Naidu
Name: K-Means Clustering
Creds: DataCamp

"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
samples = iris.data

print(samples)
model = KMeans(n_clusters = 3) #3 clusters becasue there are 3 species of iris
model.fit(samples) #fits the model to the data by locating and remembering the regions where the different clusters occur
labels = model.predict(samples) #returns a cluster lable for each sample, indicated to which cluster a sample belongs 
print(labels)
