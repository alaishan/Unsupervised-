#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:40:56 2021

@author: Alaisha Naidu
Name: t-SNE for 2D Maps
Creds: DataCamp
Dataset: Iris Flower Dataset

"""
#t-SNE = t-distributed stochastic neighbour embedding
#maps samples from a high dimensional space to a 2D/3D space so they can be visualized
#distortion exists in projection BUT distances between samples is preserved

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

iris = load_iris()
samples = iris.data
species = iris.target #list of species of each sample

model = TSNE(learning_rate=100) #learning rate - wrong rate: samples bunched together (normally between 50 and 200)

transformed = model.fit_transform(samples) #t-SNE only has a fit_transform() method - not separate. Cannot include new samples, must start again
xs = transformed[:,0]
ys = transformed[:,1]
plt.title('t-SNE applied to Iris Flower Dataset')
plt.scatter(xs, ys, c = species) # c = colour, ie colour points using the species
plt.show()

#axes of t-SNE plot have no meaning. 