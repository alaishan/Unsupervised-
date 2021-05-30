#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:49:07 2021

@author: Alaisha Naidu
Name: PCA Visualization
Creds: DataCamp, University of Cape Town - Geomatics Dept, Prof Julian Smit
Dataset: Piedmont Wines Dataset

"""
#Dimension Reduction - finds patterns in data which are used to rexpress the same data in a compressed form
#More efficient storage and computation, removes noise

#Principal Component Analysis
#Decorrelation - rotates samples to be in line with coordinate axes, shifts samples so that they have a mean 0
#Above done with the PCA fit() method learns how to shift samples, transform() method applies the learned shift


import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

wine = load_wine()
samples = wine.data

df = pd.DataFrame(samples, columns=wine.feature_names)
samples = df.loc[:,['total_phenols','od280/od315_of_diluted_wines']] #create an array with just these two columns
#print(samples)

model = PCA()
model.fit(samples)
transformed = model.transform(samples)
#print(transformed)

print(model.components_) #displays principal components of the dataset

#PCA features are not correlated
#Linear correlation is evaluated with the Pearson Correlation 
#(between -1 ie strong -ive correllation and 1 ie strong +ive correlation, 0 no correlation )


"""
Original Array:
         total_phenols  od280/od315_of_diluted_wines
0             2.80                          3.92
1             2.65                          3.40
2             2.80                          3.17
3             3.85                          3.45
4             2.80                          2.93
..             ...                           ...
173           1.68                          1.74
174           1.80                          1.56
175           1.59                          1.56
176           1.65                          1.62
177           2.05                          1.60

New Array:
[[-1.32771994e+00  4.51396070e-01]
 [-8.32496068e-01  2.33099664e-01]
 [-7.52168680e-01 -2.94789161e-02]
 [-1.64026613e+00 -6.55724013e-01]
 [-5.67992278e-01 -1.83358911e-01]
              ...              ...
 [ 1.06332236e+00 -8.68573465e-02]
 [ 1.12451466e+00 -2.94355544e-01]
 [ 1.25915966e+00 -1.33201192e-01]
 [ 1.17464556e+00 -1.40775294e-01]
 [ 9.33526935e-01 -4.60559297e-01]
 
Principal Components:
[[-0.64116665 -0.76740167]
 [-0.76740167  0.64116665]]
    
"""