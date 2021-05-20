#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:19:51 2021

@author: Alaisha Naidu
Name: Transforming Features Pipeline
Creds: DataCamp
Dataset: Piedmont Wines Dataset from Italy

"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

wine = load_wine()
samples = wine.data

scaler = StandardScaler() #gives every feature mean 0 and variance 1
kmeans = KMeans(n_clusters = 3) #3 clusters becasue there are 3 varieties of wines
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples) #fits the model to the data by locating and remembering the regions where the different clusters occur

labels = pipeline.predict(samples) #returns a cluster lable for each sample, indicated to which cluster a sample belongs 
print(kmeans.inertia_) #use after fit
#K-Means automatically generates clusters that minimizes the cluster inertia 
#print(labels)

xs = samples[:, 0] #od280 in 0th column
ys = samples[:, 2] #malic acid in 2nd column
plt.scatter(xs, ys, c = labels) #c = labels colours by cluster label
plt.show()

names = wine.target_names
df = pd.DataFrame(samples, columns=wine.feature_names)
df['varieties'] = wine.target
df['varieties'] = df['varieties'].replace(to_replace= [0, 1, 2], value = ['barbera','grignolino', 'barolo']) ### SUPER IMPORTANT - REPLACES 0,1,2 with variety name
varieties = df['varieties'].values.tolist()


#compare the species count to classification count 
#This is an Accuracy Assessment 
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
print(df)
#crosstabulation
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

#features of wine dataset have very different variances - variance of a feature measures spead of its values
#In K-Means clustering, the variance of a feature corresponds to its influence on the clustering algorithm 
#To give every feature an even chance, the data must be transformed so that the features all have an equal variance

scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled= scaler.transform(samples)

#StandSCaler transforms the data using the transform method while K-Means assigns cluster labels to samples using predict method

"""
Output: can see that the clusters are very mixed!

      labels   varieties
0         1     barbera
1         1     barbera
2         1     barbera
3         1     barbera
4         0     barbera
..      ...         ...
173       0  grignolino
174       0  grignolino
175       0  grignolino
176       0  grignolino
177       2  grignolino

Before standardization: 

[178 rows x 2 columns]
varieties  barbera  barolo  grignolino
labels                                
0               29      13          20
1                0      46           1
2               19       0          50

After standardization:

[178 rows x 2 columns]
varieties  barbera  barolo  grignolino
labels                                
0                0      59           3
1               48       0           3
2                0       0          65

"""