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
print(model.inertia_) #use after fit
#K-Means automatically generates clusters that minimizes the cluster inertia 
#print(labels)

names = iris.target_names
df = pd.DataFrame(samples, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica']) ### SUPER IMPORTANT - REPLACES 0,1,2 with sepcies name
species = df['species'].values.tolist()
#species = ['setosa', 'setosa', 'versicolor', 'virginica', 'setosa', ...] straight from the original dataset
#print(species) #list of names of species in a dataframe

#compare the species count to classification count 
#This is an Accuracy Assessment 
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
#crosstabulation
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

#Measuring the quality of the Clusterings by their Interia 
#Inertia of a clustering the the distance of samples to the centroids of the clusters
#Tighter clusters mean a more accurate classification

"""
Inertia Value:
    
78.851441426146 Three clusters is the elbow of the inertia curve 

Dataframe Output: 
    
     labels    species
0         0     setosa
1         0     setosa
2         0     setosa
3         0     setosa
4         0     setosa
..      ...        ...
145       2  virginica
146       1  virginica
147       2  virginica
148       2  virginica
149       1  virginica

[150 rows x 2 columns]


Crosstabulation Output:

[150 rows x 2 columns]
species  setosa  versicolor  virginica
labels                                
0             0          48         14
1            50           0          0
2             0           2         36
    
"""
