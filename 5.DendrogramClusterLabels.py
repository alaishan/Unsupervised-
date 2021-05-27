#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:03:56 2021

@author: Alaisha Naidu
Name: Cluster Labels in Dendrogram
Creds: DataCamp, KamKam (StackOverflow)
Dataset: Eurovision 2016 results

"""
#Hierarchical Clustering is a Visualization tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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

mergings = linkage(samples, method ='complete') #Distance between clusters.
#complete linkage method means that the distance between clusters is max. distance between their samples


#agglomerative hierarchical clustering means is a Bottom-Up clustering. With the bottom level having the most clusters
#Each country being a cluster and then joining them together semantically

dendrogram(mergings,
           labels=country_names.tolist(),
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

#The y-axis of a dendrogram shows the distance between merging clusters
labels = fcluster(mergings,30, criterion ='distance')

print(labels)

#to inspect the cluster labels, use a DataFrame to align labels with country names
pairs = pd.DataFrame({'labels':labels, 'countries':country_names})
print(pairs.sort_values('labels')) #sort by cluster label
#SciPy Cluster labels start at 1 not 0

"""
Output: 
    NumPy Array containing cluster labels for all the countries

[25 28 18  1 27 30 12 22 32 23 33 38 10  6 19  8 14 31  3 34 39 11 16 36
 29  4  4 26 37 24 10  7 35  5 20 21 15  9  2 13 30 17]

    Pandas DataFrame
    
    labels             countries
3        1               Austria
38       2           Switzerland
18       3               Germany
25       4                Latvia
26       4             Lithuania
33       5            San Marino
13       6               Estonia
31       7                Poland
15       8               Finland
37       9                Sweden
30      10                Norway
12      10               Denmark
21      11               Iceland
6       12               Belgium
39      13       The Netherlands
16      14                France
36      15                 Spain
22      16               Ireland
41      17        United Kingdom
2       18             Australia
14      19      F.Y.R. Macedonia
34      20                Serbia
35      21              Slovenia
7       22  Bosnia & Herzegovina
9       23               Croatia
29      24            Montenegro
0       25               Albania
27      26                 Malta
4       27            Azerbaijan
1       28               Armenia
24      29                 Italy
40      30               Ukraine
5       30               Belarus
17      31               Georgia
8       32              Bulgaria
10      33                Cyprus
19      34                Greece
32      35                Russia
23      36                Israel
28      37               Moldova
11      38        Czech Republic
20      39               Hungary

"""




