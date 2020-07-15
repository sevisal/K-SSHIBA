#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:05:12 2020

@author: root
"""

import pickle
import numpy as np
import sys
sys.path.append('./lib/')
import ksshiba as ksshiba

        
my_dict = pickle.load( open('../Databases/PubFig/data_lfwa_0.4.pkl', "rb" ), encoding='latin1' )


targets = ['Eyeglasses', 'Sunglasses']
idx_target_eyeglasses = list(my_dict['attributes']).index(targets[0])
idx_target_sunglasses = list(my_dict['attributes']).index(targets[1])

X = my_dict['X'].astype(float)
Y_eye = my_dict['Y'][:,idx_target_eyeglasses]
Y_sun = my_dict['Y'][:,idx_target_sunglasses]

Y_eye = np.where(Y_eye>0, 1, 0)
Y_sun = np.where(Y_sun>0, 1, 0)



h = my_dict['h']
w = my_dict['w']
target_names = my_dict['target']

X = X[:100,:]
    
idx = np.arange(X.shape[0],)
from sklearn.model_selection import train_test_split
idx_tr, idx_tst = train_test_split(idx, test_size=0.3, random_state = 31)
X_tr = X[idx_tr,:]
X_tst = X[idx_tst,:]

Y_tr = Y_eye[idx_tr, np.newaxis].astype(int)
Y_tst = Y_eye[idx_tst, np.newaxis].astype(int)

#Y_tr = Y_sun[idx_tr, np.newaxis].astype(int)
# Y_tst = Y_sun[idx_tst, np.newaxis].astype(int)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_tst = scaler.transform(X_tst)
X = np.vstack((X_tr, X_tst))
Y = np.vstack((Y_tr, Y_tst))

myKc = 100            # number of latent features
prune = 1             # whether to prune the irrelevant latent features
sparse_fs = 1         # whether or not to carry out feature selection

# Training KSSHIBA, in the dual space
myKModel = ksshiba.SSHIBA(myKc, prune)   
#Y1 = myKModel.struct_data(Y, 'cat', 0)
Y2_tr = myKModel.struct_data(Y_tr, 'mult', 0)
Y2_tst = myKModel.struct_data(Y_tst, 'mult', 0) 
K0 = myKModel.struct_data(X, 'reg', 0, V = X, kernel = 'rbf', sparse_fs = sparse_fs)

myKModel.fit(K0, Y2_tr, pruning_crit = 1e-2, max_iter = 10, Y_tst = Y2_tst, AUC = 1, verbose = 1)
print('AUC KSSHIBA: %.3f' %(myKModel.AUC[-1]))
