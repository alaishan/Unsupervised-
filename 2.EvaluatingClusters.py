#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:06:26 2021

@author: Alaisha Naidu
Name: Evaluating Clusterings 
Creds: DataCamp

"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#K-Means Clustering uses the centroid of a p-dimensional cluster to classify a sample

iris = load_iris()
samples = iris.data
#print(samples)

model = KMeans(n_clusters = 3) #3 clusters becasue there are 3 species of iris
model.fit(samples) #fits the model to the data by locating and remembering the regions where the different clusters occur
labels = model.predict(samples) #returns a cluster lable for each sample, indicated to which cluster a sample belongs 
#print(labels)

names = iris.target_names
df = pd.DataFrame(samples, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica']) ### SUPER IMPORTANT - REPLACES 0,1,2 with sepcies name
species = df['species'].values.tolist()
#species = ['setosa', 'setosa', 'versicolor', 'virginica', 'setosa', ...]
print(species) #list of names of species in a dataframe

#compare the species count to classification count 

#df = pd.DataFrame({'Lables': labels, 'Species': species})
