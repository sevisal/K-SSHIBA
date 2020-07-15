#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:16:44 2020

@author: root
"""
import numpy as np
from matplotlib import pyplot as plt
import os
dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
(prv_fold,thisfold) = os.path.split(dirpath)
os.sys.path.append(prv_fold +'/lib/')

from sparseard import SparseELBO


# ------ Initialization --------      
# Size of latent space  
k = 5
# Nº of samples
n = 100
# Nº of support vector do you want to use
n_ = 20
# Nº of features of X
d = 10
# Boolean to chose if u want feature selection
fs = True
# Learning rate to optimizer steps
lr = 1e-3
# Steps to optimize
steps = 5000
# Verbose: if u want to see lower bound evolution during training
verbose = True


# ------ CREATE THE TOY DATASET ---------
# Z is the latent-space projection of shape NxK
Z = np.random.randn(n,k)
# W is the matrix of this view of shape DxK
W = np.random.randn(d,k)
# Create the dataset as Z@W.T + 10 noise features
X = np.hstack((np.dot(Z,W.T), np.ones((n,d))))
# Create the support vectors as just a partition of X
V = np.copy(X)
V = V[:n_,:]
# Create the matrix A that is basically W but of shape V
A = np.dot(V,np.vstack((W, np.random.randn(d,k))))

# ------ TRAIN THE MODULE -------------------

sp = SparseELBO(X, V, fs=fs, lr=lr, verbose=verbose)
sp.sgd_step(Z@A.T, steps)
lengthscale, kernel, var = sp.get_params()

plt.figure()
plt.plot(sp.loss)
plt.xlabel("Epochs")
plt.ylabel("Lower bound")
plt.show()

plt.figure()
plt.stem(lengthscale)
plt.ylabel("$\lambda_d$")
plt.xlabel("Dimensions of X")
plt.xticks(range(2*d))
plt.show()