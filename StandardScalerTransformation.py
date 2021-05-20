#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:41:42 2021

@author: Alaisha Naidu 
Name: Transforming Features 
Creds: DataCamp
Dataset: Piedmont Wines

"""


from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


wine = load_wine()
samples = wine.data


#features of wine dataset have very different variances - variance of a feature measures spead of its values
#In K-Means clustering, the variance of a feature corresponds to its influence on the clustering algorithm 
#To give every feature an even chance, the data must be transformed so that the features all have an equal variance
scaler = StandardScaler() #gives every feature mean 0 and variance 1
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled= scaler.transform(samples)

#StandSCaler transforms the data using the transform method while K-Means assigns cluster labels to samples using predict method