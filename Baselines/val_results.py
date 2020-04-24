# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
from scipy.io import arff
import copy
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
my_dict['jura'] =   [3]
my_dict['wq'] =     [14]
my_dict['enb'] =    [2]
my_dict['slump'] =  [3]



with open('Paper_results.pkl', 'wb') as output:
    pickle.dump(my_dict, output, pickle.HIGHEST_PROTOCOL)
    
paper_res = pickle.load( open( 'Paper_results.pkl', "rb" ), encoding='latin1' )


def Baselines_func(folds, base, database, val=False, verbose=True ):

    # --------------------Load database------------------
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

    verboseprint = print if verbose else lambda *a, **k: None
    
    # bases = ['KPCA_LR', 'KCCA_', 'KCCA_LR', '_KRR', '_SVRrbf', '_NN'] # Name of the baseline
    bases = ['KPCA_LR', 'KCCA_LR']
    r2_final= {'KPCA_LR': [], 'KCCA_LR': []}
    latent_factors = []
    v_dim = np.arange(5,105,5)
    
    for base in bases:
        # We separate the baseline into the different options available
        pipeline = base.split('_')
        print('Training '+pipeline[0]+' FE and '+pipeline[1]+' Classifier')
        r2_total = np.zeros((len(v_dim), len(fold_tst), len(fold_tst), 5))
        lf_total = np.zeros((len(v_dim), len(fold_tst), len(fold_tst), 5))
        
        for z in range(5):
            r2_val = np.zeros((len(v_dim), len(fold_tst), len(fold_tst)))
            lf_val = np.zeros((len(v_dim), len(fold_tst), len(fold_tst)))
            for i in np.arange(len(fold_tst)):
                # At his point we check whether the file where we want to store the results does or doesn't already exist.
                # If it does we check if this baseline has been stored and, if so, we load it. If the baseline isn't in the file, we define it.
                filename = 'Results/Baselines_'+database+'_'+str(folds)+'folds.pkl'
                if os.path.exists(filename):
                    verboseprint ("Loading existing model...")
                    results = pickle.load( open( filename, "rb" ) )
                    if base in results:
                        verboseprint ("... Model loaded")
                    else:
                        results[base] = {}
                        results[base]['R2'] = np.zeros((len(fold_tst),))
                        results[base]['mse'] = np.zeros((len(fold_tst),))
                        results[base]['Kc'] = np.zeros((len(fold_tst),))
                        results[base]['gamma_used'] = np.zeros((len(fold_tst),))
                        verboseprint ("... Model defined")
                else:
                    results = {}
                    results[base] = {}
                    results[base]['R2'] = np.zeros((len(fold_tst),))
                    results[base]['mse'] = np.zeros((len(fold_tst),))
                    results[base]['Kc'] = np.zeros((len(fold_tst),))
                    results[base]['gamma_used'] = np.zeros((len(fold_tst),))
                
                verboseprint('---------> Fold '+str(i)+' <---------')   
                
                if results[base]['R2'][i] == 0.0:
                    # Splitting the data into training and test sets.
                    
                    pos_tr = fold_tst[i][0]
                    pos_tst =  fold_tst[i][1]
                    
                    Y_tr = Y[pos_tr] 
                    Y_tst = Y[pos_tst]
                    X_tr = X[pos_tr,:]
                    X_tst = X[pos_tst,:]
                    
                    #from sklearn.impute import SimpleImputer
                    #simp = SimpleImputer(missing_values=np.nan, strategy='mean')
                    #X_tr = simp.fit_transform(X_tr)
                    #X_tst = simp.transform(X_tst)
                    
                    # Normalizing the data
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_tst = scaler.transform(X_tst)
                    
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
            
                    
                    for j in np.arange(len(fold_tst)):
                        pos_tr2 = dict_fold_val[i][j][0]
                        pos_val =  dict_fold_val[i][j][1]
                        Y_val = Y_tr[pos_val]
                        Y_tr2 = Y_tr[pos_tr2]
                        X_val = X_tr[pos_val,:]
                        X_tr2 = X_tr[pos_tr2,:]
                        
                        scaler = StandardScaler()
                        X_tr2 = scaler.fit_transform(X_tr2)
                        X_val = scaler.transform(X_val)
                        
                        for p, ss in enumerate(v_dim):
                            V = np.copy(X_tr2)
                            idx = np.random.randint(V.shape[0], size=int(ss*V.shape[0]))
                            V_ss = V[idx, :]
                            
                            if pipeline[0] == 'KPCA':
                                K_tr = rbf_kernel(X_tr2, V_ss, gamma[database][0])
                                K_val = rbf_kernel(X_val, V_ss, gamma[database][0])
                                K_tr = center_K(K_tr)
                                K_val = center_K(K_val)
                                
                                pca = PCA()
                                P_tr = pca.fit_transform(K_tr)
                                P_tst = pca.transform(K_val)
                                # Selecting the latent factors that explain 95% of the variance.
                                Kc = 0
                                while np.sum(pca.explained_variance_ratio_[:Kc]) < 0.95:
                                    Kc = Kc + 1
                                P_tr = P_tr[:, :Kc]
                                P_tst = P_tst[:, :Kc]
                                lf_val[p,i,j] = Kc
                                print("Trained a KPCA_LR")
                            
                            elif pipeline[0] == 'KCCA':
                                # KCCA
                                K_tr = rbf_kernel(X_tr2, V_ss, gamma[database][2])
                                K_val = rbf_kernel(X_val, V_ss, gamma[database][2])
                                K_tr = center_K(K_tr)
                                K_val = center_K(K_val)
                                
                                cca = CCA(n_components = Y_tr.shape[1]-1).fit(K_tr, Y_tr2)
                                P_tr = cca.transform(K_tr)
                                P_tst = cca.transform(K_val)
                                #[p,i,j] = r2_score(Y_val, cca.predict(K_val), multioutput = 'uniform_average') # = 'variance_weighted') 
                                print("Trained a KCCA_LR")
                            if pipeline[1] == 'LR':
                                # Linear Regression
                                reg = LinearRegression()
                                reg.fit(P_tr, Y_tr2)
                                Y_pred = reg.predict(P_tst)
                                r2_val[p,i,j] = r2_score(Y_val, Y_pred, multioutput = 'uniform_average')
                                
                            print("------%%%%%-----")
                            print("Something trained")
                            print("i:",i)
                            print("j:",j)
                            print("p:",p)
                            print(r2_val[p,i,j])
                            print(lf_val[p,i,j])
                            print("------%%%%%-----")
                    
                    
                    verboseprint('... classifier trained.\n')
                    # Storing the results.
                    verboseprint(base + ' R2: %0.2f%%' %(results[base]['R2'][i]*100))
                    verboseprint(base + ' MSE:      %0.3f' %(results[base]['mse'][i]))
                    
                    if os.path.exists(filename):
                        results2 = pickle.load( open( filename, "rb" ) )
                        results2[base] = results[base]
                    else:
                        results2 = copy.copy(results)            
                    with open(filename, 'wb') as output:
                        pickle.dump(results2, output, pickle.HIGHEST_PROTOCOL)
                        
                else:
                    verboseprint('Fold previously trained. ' + base + ' R2: %0.3f\n                                mse: %0.3f' %(results[base]['R2'][i], results[base]['mse'][i]))
            
            r2_total[:, :, :, z] = r2_val
            lf_total[:, :, :, z] = lf_val
        r2_final[base].append(r2_total)
        #print(base +' mean R2:  %0.3f +/- %0.3f%%' %(np.mean(results[base]['R2']) , np.std(results[base]['R2'])))
        #print(base +' mean MSE: %0.3f +/- %0.3f' %(np.mean(results[base]['mse']) , np.std(results[base]['mse'])))
        #print(base +' mean Kc:  %0.3f +/- %0.3f' %(np.mean(results[base]['Kc']) , np.std(results[base]['Kc'])))
    return r2_final

#best_ss = []

database = "wq"
print("-----------------------")
print(database)
result = Baselines_func(10,'base', database, val=True)
#best_ss.append(result)
filename = database+"_r2_ss_tf_vf_5.pkl"
file1 = open("he_terminado.txt","w")

with open(filename, "wb") as output:
    pickle.dump(result, output)

# for database in paper_res:
#     print("-----------------------")
#     print(database)
#     result = Baselines_func(10,'base', database, val=True)
#     best_ss.append(result)
#     filename = database+"ss_val.pkl"
#     with open(filename, "wb") as output:
#         pickle.dump(result, output)

# filename2= "all_results_ss.pkl"
# with open(filename2, "wb") as output:
#         pickle.dump(best_ss, output)
# file1 = open("he_terminado.txt","w")
        


