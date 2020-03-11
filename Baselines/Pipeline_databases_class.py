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
def Baselines_func(folds, base):
# In[]:
    database = 'Satellite' #Here we specify the desired database
    print('Loaded database: '+database)
    
    file = 'data_'+database
    
    X = np.loadtxt(prv_fold+'/Databases/'+file+'/data.txt')
    Y = np.loadtxt(prv_fold+'/Databases/'+file+'/labels.txt')
    
    # folds = 2
    # =================================================== #
    # Don't run, just to generate folds and save in a file
    # =================================================== #
    
    # from sklearn.model_selection import StratifiedKFold
    # skf_tst = StratifiedKFold(n_splits=folds, shuffle = True)
    # fold_tst =[f for  i, f in enumerate(skf_tst.split(X, Y))]
    # dict_fold_val = {}
    # for ii, f_tst in enumerate(fold_tst):
    #     pos_tr = f_tst[0]
    #     skf_val = StratifiedKFold(n_splits=folds, shuffle = True)
    #     fold_val =[f for  i, f in enumerate(skf_val.split(X[pos_tr], Y[pos_tr]))]
    #     dict_fold_val[ii]=fold_val
    
    # pickle.dump([fold_tst, dict_fold_val], open( str(folds)+'folds_'+database+'.p', "wb" ))
    
    # =================================================== #
    
    [fold_tst, dict_fold_val] = pickle.load(open(str(folds)+'folds_'+database+'.p','rb'))

    # In[]:
    
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import StandardScaler
    
    # bases = ['KPCA_SVM', 'KCCA_', 'KCCA_SVM', '_SVMrbf'] # Name of the baseline
    # bases = ['KCCA_SVM', '_SVMrbf'] # Name of the baseline
    # We separate the baseline into the different options available
    pipeline = base.split('_')
    print('Training '+pipeline[0]+' FE and '+pipeline[1]+' Classifier')
    for i in np.arange(len(fold_tst)):
        # At his point we check whether the file where we want to store the results does or doesn't already exist.
        # If it does we check if this baseline has been stored and, if so, we load it. If the baseline isn't in the file, we define it.
        
        filename = 'Results/Baselines_'+database+'_'+str(folds)+'folds.pkl'
        if os.path.exists(filename):
            print ("Loading existing model...")
            results = pickle.load( open( filename, "rb" ) )
            if base in results:
                print ("... Model loaded")
            else:
                results[base] = {}
                results[base]['ACC'] = np.zeros((len(fold_tst),))
                results[base]['AUC'] = np.zeros((len(fold_tst),))
                results[base]['Kc'] = np.zeros((len(fold_tst),))
                print ("... Model defined")
        else:
            results = {}
            results[base] = {}
            results[base]['ACC'] = np.zeros((len(fold_tst),))
            results[base]['AUC'] = np.zeros((len(fold_tst),))
            results[base]['Kc'] = np.zeros((len(fold_tst),))
        
        print('---------> Fold '+str(i)+' <---------')   
        
        if results[base]['ACC'][i] == 0.0:
            # Splitting the data into training and test sets.
            pos_tr = fold_tst[i][0]
            pos_tst =  fold_tst[i][1]
            Y_tr = Y[pos_tr] 
            Y_tst = Y[pos_tst]
            encoder = LabelBinarizer()
            Y_tr_b = encoder.fit_transform(Y_tr)
            Y_tst_b = encoder.transform(Y_tst)
            p_class = np.sum(Y_tst_b, axis=0)/np.sum(Y_tst_b)
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
            print('Extracting features...')
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
                print('... projections defined.')
            elif pipeline[0] == 'KCCA':
                # KCCA
                cca = CCA(n_components = Y_tr_b.shape[1]-1).fit(K_tr, Y_tr_b)
                results[base]['Kc'][i] = Y_tr_b.shape[1]-1
                P_tr = cca.transform(K_tr)
                P_tst = cca.transform(K_tst)
                print('... projections defined.')
            else:
                # No feature extraction and, therefore, no kernel used.
                P_tr = np.copy(X_tr)
                P_tst = np.copy(X_tst)
                print('... no projections defined.')
            
            ############################
            # Training the classifier. #
            ############################
            print('Training the classifier...')
            if pipeline[1] == 'SVM':
                # SVM lineal
                # Hyperparameters determined using grid search 10 fold cross validation.
                grid = {"C": np.logspace(-4,4,11)}# l1 lasso l2 ridge
                clf = SVC(kernel = 'linear')
                clf_cv = GridSearchCV(clf, grid, cv=10)
                clf_cv.fit(P_tr,Y_tr)
                results[base]['ACC'][i] = clf_cv.score(P_tst,Y_tst)
                results[base]['AUC'][i] = np.sum(calcAUC(clf_cv.decision_function(P_tst), Y_tst_b)*p_class)
            elif pipeline[1] == 'SVMrbf':
                # SVM rbf, no lineal.
                # Hyperparameters determined using grid search 10 fold cross validation.
                grid = {"C": np.logspace(-4,4,11), "gamma": np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/(np.sqrt(Y_tr_b.shape[1]))}
                clf = SVC(kernel = 'rbf')
                clf_cv = GridSearchCV(clf, grid, cv=10)
                clf_cv.fit(P_tr,Y_tr)
                results[base]['ACC'][i] = clf_cv.score(P_tst,Y_tst)  
                results[base]['AUC'][i] = np.sum(calcAUC(clf_cv.decision_function(P_tst), Y_tst_b)*p_class)
            elif pipeline[1] == 'GPLVM':
                # AQUI VA EL GPLVM
                print('CAMBIAME, POR FAVOR')
            else:
                try:
                    results[base]['ACC'][i] = accuracy_score(Y_tst, np.argmax(cca.predict(K_tst),axis=1))
                    results[base]['AUC'][i] = np.sum(calcAUC(cca.predict(K_tst), Y_tst_b)*p_class)
                except:
                    print('The selected classifier is not recognised.')
            print('... classifier trained.\n')
            # Storing the results.
            print(base + ' accuracy: %0.2f%%' %(results[base]['ACC'][i]*100))
            print(base + ' AUC:      %0.3f' %(results[base]['AUC'][i]))
            
            results2 = pickle.load( open( filename, "rb" ) )
            results2[base] = results[base]
            with open(filename, 'wb') as output:
                pickle.dump(results2, output, pickle.HIGHEST_PROTOCOL)
        else:
            print('Fold previously trained. ' + base + ' accuracy: %0.2f%%\n                                       AUC: %0.3f' %(results[base]['ACC'][i]*100, results[base]['AUC'][i]))
    
    print(base +' mean accuracy: %0.2f +/- %0.2f%%' %(np.mean(results[base]['ACC']*100) , np.std(results[base]['ACC']*100)))
    print(base +' mean AUC:      %0.2f +/- %0.3f' %(np.mean(results[base]['AUC']) , np.std(results[base]['AUC'])))
    print(base +' mean Kc:       %0.2f +/- %0.2f' %(np.mean(results[base]['Kc']) , np.std(results[base]['Kc'])))

# In[]:
if __name__ == "__main__":

    from optparse import *

    parser = OptionParser()
    parser.add_option("--f", type="int", dest="folds")
    parser.add_option("--b", type="str", dest="base")

    (opt, args) = parser.parse_args()   
    Baselines_func(opt.folds, opt.base)