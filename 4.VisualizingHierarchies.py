#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:34:16 2021

@author: Alaisha Naidu
Name: Visualizing Hierarchies _ Hierarchical Clusterings
Creds: DataCamp, KamKam (StackOverflow)
Dataset: Eurovision 2016 results

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram 

eurovision = pd.read_csv('/Users/user/Desktop/eurovision16.csv')
country_names = eurovision['From country'].unique()
toCountry = eurovision['To country'].unique()
samples = np.empty((country_names.shape[0], toCountry.shape[0]))

for i in range(eurovision.shape[0]):
    toFound = False
    toCount = 0
    fromFound = False
    fromCount = 0
    while toFound == False:
        if eurovision['To country'][i] == toCountry[toCount]:
            toFound = True
            while fromFound == False:
                if eurovision['From country'][i] == country_names[fromCount]:
                    fromFound = True
                else:
                    fromCount += 1
        else:
            toCount += 1
    samples[fromCount,toCount] = eurovision['Televote Rank'][i]

samples = np.round(samples, 0)

mergings = linkage(samples, method ='complete')


#agglomerative hierarchical clustering means is a Bottom-Up clustering. With the bottom level having the most clusters
#Each country being a cluster and then joining them together semantically

dendrogram(mergings,
           labels=country_names.tolist(),
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()