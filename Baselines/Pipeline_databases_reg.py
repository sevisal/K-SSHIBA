import numpy as np
import os
import pickle
from time import time
from sklearn.metrics import roc_curve, auc, accuracy_score

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
(prv_fold,foldername) = os.path.split(dirpath)
os.sys.path.append(prv_fold +'/lib/')
os.sys.path.append(prv_fold +'\\lib\\')

# In[]:

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
    
# In[]:
# my_dict = {}
# my_dict['atp1d'] =  [6] 
# my_dict['atp7d'] =  [6] 
# my_dict['oes97'] =  [16] 
# my_dict['oes10'] =  [16] 
# my_dict['edm'] =    [2] 
# my_dict['sf1'] =    [3] 
# my_dict['sf2'] =    [3] 
# my_dict['jura'] =   [3] 
# my_dict['wq'] =     [14] 
# my_dict['enb'] =    [2] 
# my_dict['slump'] =  [3] 
# my_dict['andro'] =  [6] 
# my_dict['osales'] = [12] 
# my_dict['scfp'] =   [3] 

# with open('Paper_results.pkl', 'wb') as output:
#     pickle.dump(my_dict, output, pickle.HIGHEST_PROTOCOL)
    
paper_res = pickle.load( open( 'Paper_results.pkl', "rb" ), encoding='latin1' )

# In[]:
def Baselines_func(folds, base):
# In[]:
    # database = 'Satellite' #Here we specify the desired database
    database = 'atp7d'
    print('Loaded database: '+database)
    
    from scipy.io import arff
    data, meta = arff.loadarff(prv_fold+'/Databases/MultiTaskRegressionDatasets/' + database +'.arff')
    
    import pandas as pd
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

    # In[]:
    import copy
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    # from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score
    
    verbose = 0
    verboseprint = print if verbose else lambda *a, **k: None
    
    bases = ['KPCA_LR', 'KCCA_', 'KCCA_LR', '_KRR', '_SVMrbf', '_NN'] # Name of the baseline
    for base in bases:
        # We separate the baseline into the different options available
        pipeline = base.split('_')
        print('Training '+pipeline[0]+' FE and '+pipeline[1]+' Classifier')
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
                    verboseprint ("... Model defined")
            else:
                results = {}
                results[base] = {}
                results[base]['R2'] = np.zeros((len(fold_tst),))
                results[base]['mse'] = np.zeros((len(fold_tst),))
                results[base]['Kc'] = np.zeros((len(fold_tst),))
            
            verboseprint('---------> Fold '+str(i)+' <---------')   
            
            if results[base]['R2'][i] == 0.0:
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
        
                # Generating RBF kernel and calculating the gamma value.
                K_tr, sig = rbf_kernel_sig(X_tr, X_tr)
                K_tst, sig = rbf_kernel_sig(X_tst, X_tr, sig = sig)
                
                ##############################################
                # Defining the feature extracting algorithm. #
                ##############################################
                verboseprint('Extracting features...')
                if pipeline[0] == 'KPCA':
                    # KPCA
                    pca = PCA()
                    P_tr = pca.fit_transform(K_tr)
                    P_tst = pca.transform(K_tst)
        
                    # Selecting the latent factors that explain 95% of the variance.
                    Kc = 0
                    while np.sum(pca.explained_variance_ratio_[:Kc]) < 0.95:
                        Kc = Kc + 1 
                    results[base]['Kc'][i] = Kc
                    P_tr = P_tr[:,:Kc]
                    P_tst = P_tst[:,:Kc]      
                    verboseprint('... projections defined.')
                elif pipeline[0] == 'KCCA':
                    # KCCA
                    cca = CCA(n_components = Y_tr.shape[1]-1).fit(K_tr, Y_tr)
                    results[base]['Kc'][i] = Y_tr.shape[1]-1
                    P_tr = cca.transform(K_tr)
                    P_tst = cca.transform(K_tst)
                    verboseprint('... projections defined.')
                else:
                    # No feature extraction and, therefore, no kernel used.
                    P_tr = np.copy(X_tr)
                    P_tst = np.copy(X_tst)
                    verboseprint('... no projections defined.')
                
                ############################
                # Training the classifier. #
                ############################
                verboseprint('Training the classifier...')
                if pipeline[1] == 'LR':
                    # Linear Regression
                    reg = LinearRegression()
                    reg.fit(P_tr, Y_tr)
                    Y_pred = reg.predict(P_tst)
                    results[base]['R2'][i] = r2_score(Y_tst, Y_pred, multioutput = 'uniform_average') # = 'variance_weighted') 
                    results[base]['mse'][i] = mse(Y_tst, Y_pred, multioutput = 'uniform_average') 
                else:
                    try:
                        results[base]['R2'][i] = r2_score(Y_tst, cca.predict(K_tst), multioutput = 'uniform_average') # = 'variance_weighted') 
                        results[base]['mse'][i] = mse(Y_tst, cca.predict(K_tst), multioutput = 'uniform_average') 
                    except:
                        verboseprint('The selected classifier is not recognised.')
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
        
        print(base +' mean R2:  %0.3f +/- %0.3f%%' %(np.mean(results[base]['R2']) , np.std(results[base]['R2'])))
        print(base +' mean MSE: %0.3f +/- %0.3f' %(np.mean(results[base]['mse']) , np.std(results[base]['mse'])))
        print(base +' mean Kc:  %0.3f +/- %0.3f' %(np.mean(results[base]['Kc']) , np.std(results[base]['Kc'])))

# In[]:
if __name__ == "__main__":

    from optparse import *

    parser = OptionParser()
    parser.add_option("--f", type="int", dest="folds")
    parser.add_option("--b", type="str", dest="base")

    (opt, args) = parser.parse_args()   
    Baselines_func(opt.folds, opt.base)