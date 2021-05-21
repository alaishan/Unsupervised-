#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:34:16 2021

@author: Alaisha Naidu
Name: Visualizing Hierarchies
Creds: DataCamp
Dataset: Eurovision 2016 results

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram 

eurovision = pd.read("/Users/user/Desktop/eurovision-2016-televoting.csv")
country_names = eurovision['From country'].unique()
toCountry = eurovision['To country'].unique()
samples = np.empty((country_names.shape[0], toCountry.shape[0]))

mergings = linkage(samples, method = 'complete')