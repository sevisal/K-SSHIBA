# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
import pandas as pd
from time import time
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
from scipy.io import arff
import copy
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.kernel_approximation import Nystroem

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
(prv_fold,foldername) = os.path.split(dirpath)
os.sys.path.append(prv_fold +'/lib/')
os.sys.path.append(prv_fold +'\\lib\\')

if os.path.exists("he_terminado.txt"):
  os.remove("he_terminado.txt")
  print("Removed")
else:
  print("The script has started.")


def calcAUC(Y_pred, Y_tst):
    n_classes = Y_pred.shape[1]
    # Compute ROC curve and ROC area for each class    
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros((n_classes,1))
    for i in np.arange(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_tst[:,i], Y_pred[:,i]/n_classes)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc.flatten()

def rbf_kernel_sig(X1, X2, sig=0):
    size1 = X1.shape[0];
    size2 = X2.shape[0];
    if X1.ndim==1:
        X1 = X1[:,np.newaxis]
        X2 = X2[:,np.newaxis]
    G = (X1* X1).sum(axis=1)
    H = (X2* X2).sum(axis=1)
    Q = np.tile(G, [size2,1]).T
    R = np.tile(H, [size1,1])
    KK=np.dot(X1,X2.T)
    dist=(Q + R - 2*KK)
    if sig == 0:  # Then, we estimate its value
        aux = dist-np.tril(dist)
        aux = aux.reshape(size1**2,1)
        sig = np.sqrt(0.5*np.mean(aux[np.where(aux>0)]))             
    K = np.exp(-dist/sig**2);
    return K, sig

def center_K(K):
    """Center a kernel matrix K, i.e., removes the data mean in the feature space
    Args:
        K: kernel matrix"""
    size_1,size_2 = K.shape;
    D1 = K.sum(axis=0)/size_1
    D2 = K.sum(axis=1)/size_2
    E = D2.sum(axis=0)/size_1
    K_n = K + np.tile(E,[size_1,size_2]) - np.tile(D1,[size_1,1]) - np.tile(D2,[size_2,1]).T
    return K_n

    

my_dict = {}
my_dict['atp1d'] =  [6]
my_dict['atp7d'] =  [6]
my_dict['oes97'] =  [16]
my_dict['oes10'] =  [16]
my_dict['edm'] =    [2]
# my_dict['sf1'] =    [3]
# my_dict['sf2'] =    [3]
my_dict['jura'] =   [3]
my_dict['wq'] =     [14]
my_dict['enb'] =    [2]
my_dict['slump'] =  [3]
#my_dict['andro'] =  [6]
#my_dict['osales'] = [12]
#my_dict['scpf'] =   [3]

gamma = {}
gamma['atp1d'] =  [0.00207, 0.00159, 0.00159]
gamma['atp7d'] =  [0.00159, 0.00159, 0.00159]
gamma['oes97'] =  [0.00342, 0.00391, 0.00391]
gamma['oes10'] =  [0.00391, 0.00391, 0.00391]
gamma['edm'] =    [0.07513, 0.08839, 0.08839]
gamma['jura'] =   [0.07217, 0.07217, 0.07217]
gamma['wq'] =     [0.13363, 0.13363, 0.13363]
gamma['enb'] =    [0.08839, 0.08839, 0.08839]
gamma['slump'] =  [0.13712, 0.07217, 0.07217]


sv_opt = {}

sv_opt['KCCA_LR'] = pickle.load( open( 'sv_opt_kcca.pkl', "rb" ), encoding='latin1' ) 
sv_opt['KPCA_LR'] = pickle.load( open( 'sv_opt_kpca.pkl', "rb" ), encoding='latin1' ) 




with open('Paper_results.pkl', 'wb') as output:
    pickle.dump(my_dict, output, pickle.HIGHEST_PROTOCOL)
    
paper_res = pickle.load( open( 'Paper_results.pkl', "rb" ), encoding='latin1' )


def Baselines_func(folds, base, database, val=False):

    # database = 'Satellite' #Here we specify the desired database
    # database = 'atp1d'
    print('Loaded database: '+database)
    
    data, meta = arff.loadarff(prv_fold+'/Databases/MultiTaskRegressionDatasets/' + database +'.arff')
    df = pd.DataFrame(data)
    n_classes = paper_res[database][0]
    X = np.array(df)[:,:-n_classes].astype(float)
    Y = np.array(df)[:,-n_classes:].astype(float)
    idx = np.random.randint(0,2,Y.shape[0]).astype(int)
  
    # folds = 10
    # =================================================== #
    # Don't run, just to generate folds and save in a file
    # =================================================== #
    
    folds_file = str(folds)+'folds_'+database+'.p'
    if os.path.exists(folds_file):
        print('Loading test and validation folds.')
        [fold_tst, dict_fold_val] = pickle.load(open(folds_file,'rb'))
    else:
        print('Generating test and validation folds.\nCAREFUL, you need to always use the same partitions!')
        from sklearn.model_selection import StratifiedKFold
        skf_tst = StratifiedKFold(n_splits=folds, shuffle = True)
        # fold_tst =[f for  i, f in enumerate(skf_tst.split(X, Y))]
        fold_tst =[f for  i, f in enumerate(skf_tst.split(X, idx))]
        dict_fold_val = {}
        for ii, f_tst in enumerate(fold_tst):
            pos_tr = f_tst[0]
            skf_val = StratifiedKFold(n_splits=folds, shuffle = True)
            fold_val =[f for  i, f in enumerate(skf_val.split(X[pos_tr], idx[pos_tr]))]
            dict_fold_val[ii]=fold_val
        
        pickle.dump([fold_tst, dict_fold_val], open(folds_file, "wb" ))
    # =================================================== #
    
    verbose = 1
    verboseprint = print if verbose else lambda *a, **k: None
    
    # bases = ['KPCA_LR', 'KCCA_', 'KCCA_LR', '_KRR', '_SVRrbf', '_NN'] # Name of the baseline
    bases = ['KPCA_LR', 'KCCA_LR']
    r2_final= {'KPCA_LR': [], 'KCCA_LR': []}
    latent_factors = {'KPCA_LR': [],'KCCA_LR': []}
    v_dim= np.arange(0.05,1.05,0.05)
    R = 5
    r2_tf_ss_5 = np.zeros((len(fold_tst), R))
    fl_tf_ss_5 = np.zeros((len(fold_tst), R))
    results = {}
    for base in bases:
        # We separate the baseline into the different options available
        pipeline = base.split('_')
        print('Training '+pipeline[0]+' FE and '+pipeline[1]+' Classifier')

        results[base]= {'R2': [], 'Kc': []}
        
        for i in np.arange(len(fold_tst)):

            verboseprint('---------> Fold '+str(i)+' <---------')   
            
            
            # Splitting the data into training and test sets.
            
            pos_tr = fold_tst[i][0]
            pos_tst =  fold_tst[i][1]
            
            Y_tr = Y[pos_tr] 
            Y_tst = Y[pos_tst]
            X_tr = X[pos_tr,:]
            X_tst = X[pos_tst,:]
            
            
            # Normalizing the data
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_tst = scaler.transform(X_tst)
            V = np.copy(X_tr)
            
            

            if pipeline[0] == 'KPCA':
                sv_idx = sv_opt['KPCA_LR'][database]['KPCA_LR'][i]
                ss = v_dim[sv_idx]
                # KPCA
                # Generating RBF kernel and calculating the gamma value.
                n_comp = int(ss*V.shape[0])
                kernelizer = Nystroem(kernel="rbf", 
                                      gamma=gamma[database][0], 
                                      n_components=n_comp)
                K_tr = kernelizer.fit_transform(X_tr)
                K_tst = kernelizer.transform(X_tst)
        
                # Center the kernel.
                # K_tr = center_K(K_tr)
                # K_tst = center_K(K_tst)
                
                pca = PCA()
                P_tr = pca.fit_transform(K_tr)
                P_tst = pca.transform(K_tst)
                # Selecting the latent factors that explain 95% of the variance.
                Kc = 0
                while np.sum(pca.explained_variance_ratio_[:Kc]) < 0.95:
                    Kc = Kc + 1
                results[base]['Kc'].append(Kc)
                P_tr = P_tr[:, :Kc]
                P_tst = P_tst[:, :Kc]

                # verboseprint('... projections defined.')
            elif pipeline[0] == 'KCCA':
                sv_idx = sv_opt['KCCA_LR'][database]['KCCA_LR'][i]
                ss = v_dim[sv_idx]
                n_comp = int(ss*V.shape[0])
                if n_comp < Y_tr.shape[1]:
                    print("ESTAMOS USANDO EL MiNIMO POSIBLE")
                    n_comp = Y_tr.shape[1]
                kernelizer = Nystroem(kernel="rbf", 
                                      gamma=gamma[database][0], 
                                      n_components=n_comp)
                K_tr = kernelizer.fit_transform(X_tr)
                K_tst = kernelizer.transform(X_tst)
                # KCCA
                cca = CCA(n_components = Y_tr.shape[1]-1).fit(K_tr, Y_tr)
                results[base]['Kc'].append(Y_tr.shape[1]-1)
                P_tr = cca.transform(K_tr)
                P_tst = cca.transform(K_tst)

                verboseprint('... projections defined.')
            
            
            ############################
            # Training the classifier. #
            ############################
            verboseprint('Training the classifier...')
            if pipeline[1] == 'LR':
                # Linear Regression
                reg = LinearRegression()
                reg.fit(P_tr, Y_tr)
                Y_pred = reg.predict(P_tst)
                print("Trained a: "+str(pipeline))
                results[base]['R2'].append(r2_score(Y_tst, Y_pred, multioutput = 'uniform_average'))
                print(results[base]['R2'][i])
            print("i:"+str(i))
               
        
    return results
# if __name__ == "__main__":

#     from optparse import *

#     parser = OptionParser()
#     parser.add_option("--f", type="int", dest="folds")
#     parser.add_option("--b", type="str", dest="base")

#     (opt, args) = parser.parse_args()   
#     Baselines_func(opt.folds, opt.base)

#dbs = ["atp1d", "atp7d"]
#dbs = ["atp1d"]
#dbs= ["oes10", "oes97"]
#dbs = ["edm", "enb"]
#dbs = ["jura", "wq"]
dbs = ["atp1d", "atp7d", "oes10", "oes97", "edm", "enb", "jura", "wq"]
for database in dbs:
    print("-----------------------")
    print(database)
    r2_final = Baselines_func(10,'base', database)
    filename = database+"_opt_value.pkl"
    with open(filename, 'wb') as output:
        pickle.dump(r2_final, output, pickle.HIGHEST_PROTOCOL)
    
    
file1 = open("he_terminado.txt","w")
        
