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
    latent_factors = {'KPCA_LR': [], 'KCCA_LR': []}
    v_dim = np.arange(5,105,5)
    r2_tf_ss_5 = np.zeros((len(fold_tst), len(v_dim), 10))
    fl_tf_ss_5 = np.zeros((len(fold_tst), len(v_dim), 10))
    
    for base in bases:
        # We separate the baseline into the different options available
        pipeline = base.split('_')
        print('Training '+pipeline[0]+' FE and '+pipeline[1]+' Classifier')
        
        
        for k in range(10):
            for i in np.arange(len(fold_tst)):
                # At his point we check whether the file where we want to store the results does or doesn't already exist.
                # If it does we check if this baseline has been stored and, if so, we load it. If the baseline isn't in the file, we define it.
                filename = 'Results/Baselines_'+database+'_'+str(folds)+'Testfolds.pkl'
                # if os.path.exists(filename):
                #     verboseprint ("Loading existing model...")
                #     results = pickle.load( open( filename, "rb" ) )
                #     if base in results:
                #         verboseprint ("... Model loaded")
                #     else:
                #         results[base] = {}
                #         results[base]['R2'] = np.zeros((len(fold_tst),len(v_dim)))
                #         results[base]['mse'] = np.zeros((len(fold_tst),len(v_dim)))
                #         results[base]['Kc'] = np.zeros((len(fold_tst),len(v_dim)))
                #         verboseprint ("... Model defined")
                # else:
                results = {}
                results[base] = {}
                results[base]['R2'] = np.zeros((len(fold_tst),len(v_dim)))
                results[base]['mse'] = np.zeros((len(fold_tst),len(v_dim)))
                results[base]['Kc'] = np.zeros((len(fold_tst),len(v_dim)))
                
                verboseprint('---------> Fold '+str(i)+' <---------')   
                
                if results[base]['R2'][i, 0] == 0.0:
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
                    
                    for p, ss in enumerate(v_dim):
                        V = np.copy(X_tr)
                        idx = np.random.randint(V.shape[0], size=int(ss*V.shape[0]))
                        V_ss = V[idx, :]
                        ##############################################
                        # Defining the feature extracting algorithm. #
                        ##############################################
        
                        if pipeline[0] == 'KPCA':
                            # KPCA
                            # Generating RBF kernel and calculating the gamma value.
                            sig = 1/np.sqrt(2*gamma[database][0])
                            K_tr, sig = rbf_kernel_sig(X_tr, V_ss, sig=sig)
                            K_tst, sig = rbf_kernel_sig(X_tst, V_ss, sig = sig)
                            
                            # Center the kernel.
                            K_tr = center_K(K_tr)
                            K_tst = center_K(K_tst)
                            
                            pca = PCA()
                            P_tr = pca.fit_transform(K_tr)
                            P_tst = pca.transform(K_tst)
                            # Selecting the latent factors that explain 95% of the variance.
                            Kc = 0
                            while np.sum(pca.explained_variance_ratio_[:Kc]) < 0.95:
                                Kc = Kc + 1
                            results[base]['Kc'][i, p] = Kc
                            P_tr = P_tr[:, :Kc]
                            P_tst = P_tst[:, :Kc]

                            verboseprint('... projections defined.')
                        elif pipeline[0] == 'KCCA':
                            sig = 1/np.sqrt(2*gamma[database][2])
                            K_tr, sig = rbf_kernel_sig(X_tr, V_ss, sig=sig)
                            K_tst, sig = rbf_kernel_sig(X_tst, V_ss, sig = sig)
                            
                            # Center the kernel.
                            K_tr = center_K(K_tr)
                            K_tst = center_K(K_tst)
                            
                            # KCCA
                            cca = CCA(n_components = Y_tr.shape[1]-1).fit(K_tr, Y_tr)
                            results[base]['Kc'][i, p] = Y_tr.shape[1]-1
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
                            results[base]['R2'][i, p] = r2_score(Y_tst, Y_pred, multioutput = 'uniform_average') # = 'variance_weighted') 
                            print(results[base]['R2'][i, p])
                            results[base]['mse'][i, p] = mse(Y_tst, Y_pred, multioutput = 'uniform_average') 
                        print("k:"+str(k)+"| i:"+str(i)+"|p:"+str(p))
                        # elif pipeline[1] == 'SVRrbf':
                        #     # SVM rbf, no lineal.
                        #     # Hyperparameters determined using grid search 10 fold cross validation.
                        #     r2_tmp = []
                        #     mse_tmp = []
                        #     for output in np.arange(Y_tr.shape[1]):
                        #         grid = {"C": np.logspace(-4,4,11), "gamma": np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/(np.sqrt(Y_tr.shape[1]))}
                        #         clf = SVR(kernel = 'rbf')
                        #         clf_cv = GridSearchCV(clf, grid, cv=10, n_jobs=-1)
                        #         clf_cv.fit(P_tr, Y_tr[:, output])
                        #         r2_tmp.append(r2_score(Y_tst[:, output][:, np.newaxis], clf_cv.predict(P_tst)))
                        #         mse_tmp.append(mse(Y_tst[:, output][:, np.newaxis], clf_cv.predict(P_tst)))
                        #     results[base]['R2'][i] = np.mean(r2_tmp) # = 'variance_weighted') 
                        #     results[base]['mse'][i] = np.mean(mse_tmp)
                        # elif pipeline[1] == 'KRR':
                        #     # SVM rbf, no lineal.
                        #     # Hyperparameters determined using grid search 10 fold cross validation.
                        #     grid = {"alpha": np.logspace(-2*10,2,11), "gamma": np.logspace(-2*10,2,11)/(np.sqrt(Y_tr.shape[1]))}
                        #     clf = KernelRidge(kernel = 'rbf')
                        #     clf_cv = GridSearchCV(clf, grid, cv=10)
                        #     clf_cv.fit(P_tr,Y_tr)
                        #     results[base]['params'][i] = clf_cv.best_params_
                        #     results[base]['R2'][i] = r2_score(Y_tst, clf_cv.predict(P_tst), multioutput = 'uniform_average') # = 'variance_weighted') 
                        #     results[base]['mse'][i] = mse(Y_tst, clf_cv.predict(P_tst), multioutput = 'uniform_average') 
                        # elif pipeline[1] == 'NN':
                        #     grid = {"hidden_layer_sizes": [(20,), (35,), (50,), (100), (75),],
                        #             "solver": ["adam"],
                        #             }
                        #     clf = MLPRegressor(max_iter=1000)
                        #     clf_cv = GridSearchCV(clf, grid, cv=10, n_jobs=-1, scoring='r2', vebose=1)
                        #     clf_cv.fit(P_tr,Y_tr)
                        #     results[base]['R2'][i] = r2_score(Y_tst, clf_cv.predict(P_tst), multioutput = 'uniform_average') # = 'variance_weighted') 
                        #     results[base]['mse'][i] = mse(Y_tst, clf_cv.predict(P_tst), multioutput = 'uniform_average') 
                        # else:
                        #     try:
                        #         results[base]['R2'][i] = r2_score(Y_tst, cca.predict(K_tst), multioutput = 'uniform_average') # = 'variance_weighted') 
                        #         results[base]['mse'][i] = mse(Y_tst, cca.predict(K_tst), multioutput = 'uniform_average') 
                        #     except:
                        #         verboseprint('The selected classifier is not recognised.')
                        verboseprint('... classifier trained.\n')
                        # Storing the results.
                        # verboseprint(base + ' R2: %0.2f%%' %(results[base]['R2'][i]*100))
                        # verboseprint(base + ' MSE:      %0.3f' %(results[base]['mse'][i]))
                    
                    if os.path.exists(filename):
                        results2 = pickle.load( open( filename, "rb" ) )
                        results2[base] = results[base]
                    else:
                        results2 = copy.copy(results)            
                    with open(filename, 'wb') as output:
                        pickle.dump(results2, output, pickle.HIGHEST_PROTOCOL)
                        
               # else:
             #       verboseprint('Fold previously trained. ' + base + ' R2: %0.3f\n                                mse: %0.3f' %(results[base]['R2'][i], results[base]['mse'][i]))
            
            r2_tf_ss_5[:, :, k] = results[base]['R2']
            fl_tf_ss_5[:, :, k] = results[base]['Kc']
            #print(base +' mean R2:  %0.3f +/- %0.3f%%' %(np.mean(results[base]['R2']) , np.std(results[base]['R2'])))
            #print(base +' mean MSE: %0.3f +/- %0.3f' %(np.mean(results[base]['mse']) , np.std(results[base]['mse'])))
            #print(base +' mean Kc:  %0.3f +/- %0.3f' %(np.mean(results[base]['Kc']) , np.std(results[base]['Kc'])))
        r2_final[base].append(r2_tf_ss_5)
        latent_factors[base].append(fl_tf_ss_5)
        
    return r2_final, latent_factors
# if __name__ == "__main__":

#     from optparse import *

#     parser = OptionParser()
#     parser.add_option("--f", type="int", dest="folds")
#     parser.add_option("--b", type="str", dest="base")

#     (opt, args) = parser.parse_args()   
#     Baselines_func(opt.folds, opt.base)
    
dbs = ["edm", "jura", "slump", "oes10", "wq", "enb"]
for database in dbs:
    print("-----------------------")
    print(database)
    r2_final, latent_factors = Baselines_func(10,'base', database)
    filename = database+"_kpca_test.pkl"
    with open(filename, 'wb') as output:
        pickle.dump([r2_final, latent_factors], output, pickle.HIGHEST_PROTOCOL)
    
    
file1 = open("he_terminado.txt","w")
        
