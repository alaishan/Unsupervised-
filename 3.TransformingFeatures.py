#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:19:51 2021

@author: Alaisha Naidu
Name: Transforming Features 
Creds: DataCamp
Dataset: Piedmont Wines Dataset from Italy

"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wine = load_wine()
samples = wine.data

model = KMeans(n_clusters = 3) #3 clusters becasue there are 3 varieties of wines
model.fit(samples) #fits the model to the data by locating and remembering the regions where the different clusters occur
labels = model.predict(samples) #returns a cluster lable for each sample, indicated to which cluster a sample belongs 
print(model.inertia_) #use after fit
#K-Means automatically generates clusters that minimizes the cluster inertia 
#print(labels)

xs = samples[:, 0] #sepal length in 0th column
ys = samples[:, 2] #petal length in 2nd column
plt.scatter(xs, ys, c = labels) #c = labels colours by cluster label
plt.show()

names = wine.target_names
df = pd.DataFrame(samples, columns=wine.feature_names)
df['varieties'] = wine.target
df['varieties'] = df['varieties'].replace(to_replace= [0, 1, 2], value = ['barbera', 'barolo', 'grignolino']) ### SUPER IMPORTANT - REPLACES 0,1,2 with variety name
varieties = df['varieties'].values.tolist()


#compare the species count to classification count 
#This is an Accuracy Assessment 
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
print(df)
#crosstabulation
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

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

[178 rows x 2 columns]
varieties  barbera  barolo  grignolino
labels                                
0               13      20          29
1               46       1           0
2                0      50          19

"""