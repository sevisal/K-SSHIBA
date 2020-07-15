import copy
import math
import sys

import numpy as np
from scipy import linalg
from scipy.stats import norm

from sparseard import SparseELBO
from qdist import Qdistribution

from sklearn.preprocessing import label_binarize
from sklearn.metrics import hamming_loss, roc_curve, auc, accuracy_score, r2_score



class SSHIBA(object):    
    """ Bayessian Inter-Battery Factor Analysis
    SSHIBA method for regression and classification, including multilabel and 
    categorical. It can function in both a supervised and semisupervised way. 
    At the same time, feature sparsity can be specified.
    
    This class estimates the parameters using the mean field approximation on 
    the graphical model.

    Parameters
    ----------
    __Kc : int, (default 2).
        number of components to extract.
    __prune : bool, (default 0).
        whether the pruning is used or not to remove the latent factors that 
        are not relevant.
    __hyper : list, (default None).
        hyperparameters used for the model.    
    __X_init : dict, (default None).
        Initialization of the variable X.
    __Z_init : dict, (default None).
        Initialization of the variable Z.
    __W_init : dict, (default [None]).
        Initialization of the variable W.
    __alpha_init : dict, (default None).
        Initialization of the variable alpha.
    __tau_init : dict, (default None).
        Initialization of the variable tau.
    __gamma_init : dict, (default None).
        Initialization of the variable gamma.
    
    Attributes
    ----------

    Example 1
    --------
    >>> import sshiba
    >>> model = sshiba.SSHIBA(5)
    >>> X0 = myModel_ml.struct_data(X, 0, 0)
    >>> X1 = myModel_ml.struct_data(Y, 0, 0)
    >>> X0_2 = myModel_ml.struct_data(X_tst, 0, 0)
    >>> myModel.fit(X0, X1, max_iter = 100)
    >>> prediction = myModel.predict([0], 1, X0_2)
    
    """
    
    def __init__(self, Kc = 2, prune = 0, fs = 0, SS_sep = 0,  hyper = None, X_init = None, 
                 Z_init = None, W_init = [None], alpha_init = None, 
                 tau_init = None, gamma_init = None, area_mask = None):
        self.Kc = int(Kc) # Number of  latent variables
        self.prune = int(prune) # Indicates whether the pruning is to be done        
        self.fs = int(fs) # Indicates whether the feature selection is to be done
        self.SS_sep = SS_sep
        self.hyper = hyper 
        self.X_init = X_init
        self.Z_init = Z_init
        self.W_init = W_init
        self.alpha_init = alpha_init
        self.tau_init = tau_init
        self.gamma_init = gamma_init

    def fit(self, *args,**kwargs):
        """Fit model to data.
        
        Parameters
        ----------
        __X: dict, ('data', 'method', 'sparse').
            Dictionary with the information of the input views. Where 'data' 
            stores the matrix with the data. These matrices have size n_samples
            and can have different number of features. If one view has a number
            of samples smaller than the rest, these values are infered assuming
            it is a semisupervised scheme. This dictionary can be built using 
            the function "struct_data".
        __max_iter: int, (default 500),
            Maximum number of iterations done if not converged.
        __conv_crit: float, (default 1e-6).
            Convergence criteria for the lower bound.
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
            
        """
        
        
        n = []
        for (m,arg) in enumerate(args):
            n.append(int(arg['data'].shape[0]))
        self.n_max = np.max(n)
        
        self.n = []
        self.d = []
        self.sparse = []
        self.sparse_fs = []
        self.center = []
        self.method = []
        self.X = []
        self.t = {}
        self.SS = []
        self.SS_mask = {}   
        self.area_mask = []
        self.V = {}
        self.k = {}
        self.sig = {}

        m = -1 
        for arg in args:
            m += 1
            if not (None in arg['data']):
                self.n.append(int(arg['data'].shape[0]))
                if arg['method'] == 'reg':   #Regression
                    self.d.append(int(arg['data'].shape[1]))
                elif arg['method'] == 'cat': #Categorical
                    self.d.append(int(len(np.unique(arg['data'][~np.isnan(arg['data'])]))))
                    # if len(arg['data'].shape) < 2:
                    #     arg['data'] = arg['data'][:, np.newaxis]
                elif arg['method'] == 'mult': #Multilabel
                    if len(arg['data'].shape) < 2:
                        arg['data'] = arg['data'][:, np.newaxis]
                    self.d.append(int(arg['data'].shape[1]))
            elif not (None in self.W_init):
                self.n.append(0)
                self.d.append(self.W_init[m]['mean'].shape[0])  
            self.sparse.append(arg['sparse'])
            self.sparse_fs.append(arg['sparse_fs'])
            self.center.append(arg['center'])
            self.method.append(arg['method'])
            self.area_mask.append(arg['mask'])

            mn = np.random.normal(0.0, 1.0, self.n_max * self.d[m]).reshape(self.n_max, self.d[m])
            info = {
                "data":     np.random.normal(0.0, 1.0, self.n_max * self.d[m]).reshape(self.n_max, self.d[m]),
                "mean":     mn,
                "cov":      mn**2 ,
                "prodT":    np.dot(mn.T,mn),
                "LH":       0,
                "Elogp":    0,
                "sumlogdet":    0,
            }
            self.X.append(info) 

            #Kernel 
            data = copy.deepcopy(arg['data'])
            if not (arg['SV'] is None):
                self.V[m] = copy.deepcopy(arg['SV'])
                self.X[m]['X'] = copy.deepcopy(arg['data'])
                self.k[m] = copy.deepcopy(arg['kernel'])
                self.sig[m] = copy.deepcopy(arg['sig'])
                #Linear Kernel
                if self.k[m] == 'linear':
                    data = np.dot(self.X[m]['X'], self.V[m].T)
                #RBF Kernel
                elif self.k[m] == 'rbf': 
                    if self.sig[m] == 'auto':
                        self.sparse_K = SparseELBO(self.X[m]['X'], self.V[m], self.sparse_fs[m])
                        _, data, var = self.sparse_K.get_params()
                    else:
                        data, self.sig[m] = self.rbf_kernel_sig(self.X[m]['X'], self.V[m], sig = self.sig[m])
                else:
                    print('Error, selected kernel doesn\'t exist')
                if self.center[m]:
                    data = self.center_K(data)
                self.d[m] = self.V[m].shape[0]
                
            #Regression    
            if arg['method'] == 'reg':   
                # self.t.append(np.ones((self.n_max,)).astype(int))
                self.X[m]['data'] = data
                self.X[m]['mean'] = np.copy(self.X[m]['data'])
                if self.n_max > self.n[m]:
                    self.X[m]['mean'] = np.vstack((self.X[m]['mean'],np.NaN * np.ones((self.n_max - self.n[m], self.d[m]))))
                self.SS_mask[m] = np.isnan(self.X[m]['mean'])
                #SemiSupervised
                if np.sum(self.SS_mask[m]) > 0:   
                    self.SS.append(True)
                   #If the matrix is preinitialised
                    if (self.X_init is None):
                        for d in np.arange(self.d[m]):
                            if np.sum(self.SS_mask[m][:,d]) == self.SS_mask[m].shape[0]:

                                self.X[m]['mean'][self.SS_mask[m][:,d],d]= np.random.normal(0.0, 1.0, np.sum(self.SS_mask[m][:,d]))
                            else:
                                self.X[m]['mean'][self.SS_mask[m][:,d],d] = np.random.normal(np.nanmean(self.X[m]['mean'][:,d],axis=0), np.nanstd(self.X[m]['mean'][:,d],axis=0), np.sum(self.SS_mask[m][:,d]))
                    else:                        
                        self.X[m]['mean'][self.SS_mask[m]]  = self.X_init[m]['mean'][self.SS_mask[m]] 
                    self.X[m]['cov'][~self.SS_mask[m]] = np.zeros((np.sum(~self.SS_mask[m]),))
                else:
                    self.SS.append(False)
                    self.X[m]['cov'] = np.zeros((self.X[m]['mean'].shape))

            #Categorical
            elif arg['method'] == 'cat':
                self.t[m] = {
                    "data":     np.squeeze(data), 
                    "mean":     (np.random.randint(self.d[m], size=[self.n_max,])).astype(float),
                }
                if self.n_max > self.n[m]:
                    self.t[m]['data'] = np.hstack((self.t[m]['data'], np.NaN * np.ones((self.n_max - self.n[m],))))
                self.SS_mask[m] = np.isnan(self.t[m]['data'])
                if np.sum(self.SS_mask[m]) > 0:
                    self.SS.append(True)
                    self.t[m]['mean'] = (np.random.randint(self.d[m], size=[self.n_max, ])).astype(float)
                    self.t[m]['mean'][~self.SS_mask[m]]  = (self.t[m]['data'][~self.SS_mask[m]] ).astype(float)
                    self.X[m]['mean'][~self.SS_mask[m]] = label_binarize(self.t[m]['mean'][~self.SS_mask[m]], classes = np.arange(self.n_max,self.d[m])).astype(float)
                    self.X[m]['cov'][~self.SS_mask[m]] = 0
                else:
                    self.SS.append(False)
                    self.t[m]['mean'] = np.copy(self.t[m]['data'])
                    self.t[m]['cov'] = np.zeros((self.t[m]['mean'].shape))
                    self.X[m]['mean'] = np.random.normal(0.0, 1.0, self.n_max * self.d[m]).reshape(self.n_max, self.d[m])
                    self.X[m]['cov'] = np.abs(np.random.normal(0.0, 1.0, self.n_max * self.d[m]).reshape(self.n_max, self.d[m]))

            #Multilabel
            elif arg['method'] == 'mult': 
                self.t[m] = copy.deepcopy(info)
                self.t[m]['data'] = data   
                if self.n_max > self.n[m]:
                    self.t[m]['data'] = np.vstack((self.t[m]['data'],np.NaN * np.ones((self.n_max - self.n[m], self.d[m]))))
                self.SS_mask[m] = np.isnan(self.t[m]['data'])
                #Initialization of X using t (The values of t are set to 0.95 and 0.05 to avoid computational problems)
                self.X[m]['mean'][~self.SS_mask[m]] = np.log(np.abs((self.t[m]['data'][~self.SS_mask[m]]).astype(float)-0.05)/(1 - np.abs((self.t[m]['data'][~self.SS_mask[m]]).astype(float)-0.05)))
                self.X[m]['cov'][~self.SS_mask[m]] = 0
                for d in np.arange(self.d[m]):
                    self.X[m]['mean'][self.SS_mask[m][:,d],d] = np.random.normal(np.nanmean(self.X[m]['mean'][~self.SS_mask[m][:,d],d],axis=0), np.nanstd(self.X[m]['mean'][~self.SS_mask[m][:,d],d],axis=0), np.sum(self.SS_mask[m][:,d]))
                    self.X[m]['cov'][self.SS_mask[m][:,d],d] = np.abs(np.random.normal(np.nanmean(self.X[m]['mean'][~self.SS_mask[m][:,d],d],axis=0), np.nanstd(self.X[m]['mean'][~self.SS_mask[m][:,d],d],axis=0), np.sum(self.SS_mask[m][:,d])))

                #SemiSupervised
                if np.sum(self.SS_mask[m]) > 0:   
                    self.SS.append(True)
                    for d in np.arange(self.d[m]):
                        self.t[m]['mean'][self.SS_mask[m][:,d],d] = np.random.normal(np.nanmean(self.t[m]['mean'][:,d],axis=0), np.nanstd(self.t[m]['mean'][:,d],axis=0), np.sum(self.SS_mask[m][:,d]))

                    self.t[m]['mean'][~self.SS_mask[m]]  = (self.t[m]['data'][~self.SS_mask[m]] ).astype(float)
                    self.t[m]['cov'][~self.SS_mask[m]]  = 0

                    # Multilabel semisupervised independent
                    m_prev = int(np.copy(m))
                    nans_t = np.sum(np.isnan(self.t[m]['data'][:self.n[m],:]), axis=0)
                    if any(nans_t != 0) and self.SS_sep:
                        a = 1 if np.sum(nans_t != 0) != 0 else 0 # If there is no view without nan we keep the first in the original view
                        for v in np.arange(a, self.d[m])[nans_t[a:] != 0]:
                            m += 1
                            self.sparse.append(arg['sparse'])
                            self.method.append(arg['method'])
                            self.n.append(int(data.shape[0]))
                            self.d.append(int(1))
                            self.SS.append(True)
                            
                            self.X.append(copy.deepcopy(info))
                            self.t.append(copy.deepcopy(self.t[m_prev]))
                            self.t[m]['data'] = self.t[m_prev]['data'][:,v,np.newaxis]
                            self.t[m]['mean'] = self.t[m_prev]['mean'][:,v,np.newaxis]
                            self.t[m]['cov'] = self.t[m_prev]['cov'][:,v,np.newaxis]
                            self.X[m]['data'] = self.X[m_prev]['data'][:,v,np.newaxis]
                            self.X[m]['mean'] = self.X[m_prev]['mean'][:,v,np.newaxis]
                            self.X[m]['cov'] = self.X[m_prev]['cov'][:,v,np.newaxis]
                            self.SS_mask[m] = np.isnan(self.t[m]['data'])

                        if a: 
                            self.d[m_prev] = int(1)
                            self.t[m_prev]['data'] = self.t[m_prev]['data'][:,0,np.newaxis]
                            self.t[m_prev]['mean'] = self.t[m_prev]['mean'][:,0,np.newaxis]
                            self.t[m_prev]['cov'] = self.t[m_prev]['cov'][:,0,np.newaxis]
                            self.X[m_prev]['data'] = self.X[m_prev]['data'][:,0,np.newaxis]
                            self.X[m_prev]['mean'] = self.X[m_prev]['mean'][:,0,np.newaxis]
                            self.X[m_prev]['cov'] = self.X[m_prev]['cov'][:,0,np.newaxis]
                            self.SS_mask[m_prev] = self.SS_mask[m_prev][:,0,np.newaxis]
                        else:     
                            self.d[m_prev] = int(np.sum(nans_t == 0))
                            self.t[m_prev]['data'] = self.t[m_prev]['data'][:,nans_t == 0]
                            self.t[m_prev]['mean'] = self.t[m_prev]['mean'][:,nans_t == 0]
                            self.t[m_prev]['cov'] = self.t[m_prev]['cov'][:,nans_t == 0]
                            self.X[m_prev]['data'] = self.X[m_prev]['data'][:,nans_t == 0]
                            self.X[m_prev]['mean'] = self.X[m_prev]['mean'][:,nans_t == 0]
                            self.X[m_prev]['cov'] = self.X[m_prev]['cov'][:,nans_t == 0]
                            self.SS_mask[m_prev] = self.SS_mask[m_prev][:,nans_t == 0]
                        
                    #If the matrix is preinitialised
                    if not(self.X_init is None): 
                        #If only the supervised part of the matrix is preinitialised
                        if self.X_init[m]['mean'].shape[0]<self.n_max: 
                            self.X[m]['mean'][~self.SS_mask[m]]  = self.X_init[m]['mean'][~self.SS_mask[m]] 
                            self.X[m]['cov'][~self.SS_mask[m]]  = self.X_init[m]['cov'][~self.SS_mask[m]] 
                        #If all the matrix is preinitialised          
                        else: 
                            self.X[m]['mean'][~self.SS_mask[m]]  = self.X_init[m]['mean'][~self.SS_mask[m]] 
                            self.X[m]['cov'][~self.SS_mask[m]]  = self.X_init[m]['cov'][~self.SS_mask[m]] 
                else:
                    self.SS.append(False)
                    self.t[m]['mean'] = np.copy(self.t[m]['data']).astype(float)
                    self.t[m]['cov'] = np.zeros(self.t[m]['mean'].shape)
                    if not(self.X_init is None):
                        if np.max(self.X_init[m]['mean']) == 1 and np.min(self.X_init[m]['mean']) == 0:
                            self.X[m]['mean'] = (2*self.X_init[m]['mean']-1).astype(float) #We cast the variables to a float in case the values used
                        else:
                            self.X[m]['mean'] = self.X_init[m]['mean'].astype(float) #We cast the variable to a float in case the values used
                        self.X[m]['cov'] = self.X_init[m]['cov'].astype(float) #We cast the variable to a float in case the values used

        self.L = []
        self.mse = []
        self.R2 = []
        self.HL = []
        self.AUC = []
        self.AUC_tr = []
        self.ACC = []
        self.ACC_tr = []
        
        self.m = m+1
        if self.hyper == None:
            self.hyper = HyperParameters(self.m)
            
        if not (None in self.W_init):
            self.Kc = self.W_init[0]['mean'].shape[1]
 
        self.q_dist = Qdistribution(self.X, self.n, self.n_max, self.d, self.Kc, self.m, self.sparse, self.method, self.SS, 
                                    self.SS_mask, self.area_mask, self.hyper, 
                                    Z_init = self.Z_init, W_init = self.W_init, alpha_init = self.alpha_init, 
                                    tau_init = self.tau_init, gamma_init = self.gamma_init)               
        self.fit_iterate(**kwargs)
    
    def fit_iterate(self, max_iter = int(1e3), pruning_crit = 1e-6, tol = 1e-3, feat_crit = 1e-6, perc = False, verbose = 0, Y_tst = [None], 
                    X_tst = [None], X_tr = [None], HL = 0, AUC = 0, ACC= 0, mse = 0, R2 = 0, m_in = [0], m_out = 1):
        """Iterate to fit model to data.
        
        Parameters
        ----------
        __max_iter: int, (default 500),
            Maximum number of iterations done if not converged.
        __conv_crit: float, (default 1e-6).
            Convergence criteria for the lower bound.
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        """
        verboseprint = print if verbose else lambda *a, **k: None
        q = self.q_dist
        for i in range(max_iter):            
            # Update the variables of the model
            self.update(Y_tst, X_tst, X_tr, HL, AUC, ACC, mse, R2, m_in, m_out)
            # Pruning if specified after each iteration
            if self.prune:
                self.depruning(1e-15)
                self.L.append(self.update_bound())
                self.pruning(pruning_crit)
                if q.Kc == 0:
                    print('\nThere are no representative latent factors, no structure found in the data.')
                    return
            else:
                self.L.append(self.update_bound())   
            #Feature selection if specified after each iteration
            if any(self.sparse):
                if self.fs:
                    self.feature_selection(feat_crit, perc)
            if verbose:
                verboseprint('\rIteration %d Lower Bound %.1f K %4d' %(len(self.L),self.L[-1], q.Kc), end='\r', flush=True)
            # Lower Bound convergence criteria
            if (len(self.L) > 100) and (abs(1 - np.mean(self.L[-101:-1])/self.L[-1]) < tol):
                verboseprint('\nModel correctly trained. Convergence achieved')             
                return
        verboseprint('')
        
    def rbf_kernel_sig(self, X1, X2, sig=0):
        """RBF Kernel.
            
        Calculates the RBF Kernel between the two different matrices. If a sig
        is not given (sig = 0) we calculate the value of sig.
        
        """
        size1 = X1.shape[0];
        size2 = X2.shape[0];
        if X1.ndim == 1:
            X1 = X1[:,np.newaxis]
            X2 = X2[:,np.newaxis]
        G = (X1* X1).sum(axis=1)
        H = (X2* X2).sum(axis=1)
        Q = np.tile(G, [size2,1]).T
        R = np.tile(H, [size1,1])
        KK = np.dot(X1,X2.T)
        dist = (Q + R - 2*KK)
        if sig == 0:  # Then, we calculate its value
            aux = (dist-np.tril(dist)).reshape(size1*size2,1)
            sig = np.sqrt(0.5*np.mean(aux[np.where(aux>0)]))             
        K = np.exp(-dist/sig**2);
        return K, sig
    
    def center_K(self, K):
        """Center a kernel matrix K, i.e., removes the data mean in the feature space
        Args:
            K: kernel matrix
        """
            
        size_1,size_2 = K.shape;
        D1 = K.sum(axis=0)/size_1
        D2 = K.sum(axis=1)/size_2
        E = D2.sum(axis=0)/size_1
        K_n = K + np.tile(E,[size_1,size_2]) - np.tile(D1,[size_1,1]) - np.tile(D2,[size_2,1]).T
        return K_n

    def pruning(self, pruning_crit):
        """Pruning of the latent variables.
            
        Checks the values of the projection matrices W and keeps the latent 
        variables if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        """
        
        q = self.q_dist
        fact_sel = np.array([])
        for m in np.arange(self.m):
            for K in np.arange(q.Kc):
                if any(abs(q.W[m]['mean'][:,K])>pruning_crit):
                    fact_sel = np.append(fact_sel,K)
        fact_sel = np.unique(fact_sel).astype(int)
        # Pruning Z
        q.Z['mean'] = q.Z['mean'][:,fact_sel]
        q.Z['cov'] = q.Z['cov'][fact_sel,:][:,fact_sel]
        q.Z['prodT'] = q.Z['prodT'][fact_sel,:][:,fact_sel]            
         # Pruning W and alpha
        for m in np.arange(self.m):
            q.W[m]['mean'] = q.W[m]['mean'][:,fact_sel]
            q.W[m]['cov'] = q.W[m]['cov'][fact_sel,:][:,fact_sel]
            q.W[m]['prodT'] = q.W[m]['prodT'][fact_sel,:][:,fact_sel]   
            q.alpha[m]['b'] = q.alpha[m]['b'][fact_sel]
        q.Kc = len(fact_sel)    
        
    def depruning(self, pruning_crit):
        """Pruning of the latent variables.
            
        Checks the values of the projection matrices W and keeps the latent 
        variables if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        """
        
        q = self.q_dist
        K_prune = self.Kc - q.Kc
        q.Z['mean'] = np.hstack((q.Z['mean'], pruning_crit*np.ones((self.n_max, K_prune))))
        q.Z['cov'] = np.vstack((np.hstack((q.Z['cov'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
        q.Z['prodT'] = np.vstack((np.hstack((q.Z['prodT'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
         # Pruning W and alpha
        for m in np.arange(self.m):
            q.W[m]['mean'] = np.hstack((q.W[m]['mean'], pruning_crit*np.ones((self.d[m], K_prune))))
            q.W[m]['cov'] = np.vstack((np.hstack((q.W[m]['cov'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
            q.W[m]['prodT'] = np.vstack((np.hstack((q.W[m]['prodT'], pruning_crit*np.ones((q.Kc, K_prune)))),pruning_crit*np.ones((K_prune, self.Kc))))
            q.alpha[m]['b'] = np.hstack((q.alpha[m]['b'], pruning_crit*np.ones((K_prune,))))
            
    def feature_selection(self, feat_crit, perc = False):
        """Feature selection.
            
        Checks the values of the projection matrices W and keeps the features
        if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        Parameters
        ----------
        __feat_crit: float.
            Indicates the feature selection criteria to follow.
        __perc: bool.
            indicates whether the value specified is a threshold or a percentage of features.
            By default it is set to work with a threshold.
        """
        
        q = self.q_dist
        feat_sel = {}
        if perc:
            for m in np.arange(self.m):
                if self.sparse[m]:
                    pos = np.argsort(q.gamma_mean(m))#[::-1]
                    feat_sel[m] = pos[:int(round(self.d[m]*feat_crit))]
            
        else:            
            for m in np.arange(self.m):
                feat_sel[m] = []
                if self.sparse[m]:
                    for d in np.arange(self.d[m]):
                        if any(abs(q.W[m]['mean'][d,:])<feat_crit):
                            feat_sel[m] = np.append(feat_sel[m],d).astype(int)
        # FS W and gamma
        for m in np.arange(self.m):
            if self.sparse[m]:
                self.X[m]['mean'] = self.X[m]['mean'][:,feat_sel[m]]
                self.X[m]['cov'] = self.X[m]['cov'][:,feat_sel[m]]
                if self.SS[m]:
                    self.SS_mask[m] = self.SS_mask[m][:,feat_sel[m]]
                    q.XS[m]['mean'] = q.XS[m]['mean'][:,feat_sel[m]]
                    q.XS[m]['cov'] = q.XS[m]['cov'][:,feat_sel[m]]
                q.W[m]['mean'] = q.W[m]['mean'][feat_sel[m],:]
                q.gamma[m]['b'] = q.gamma[m]['b'][feat_sel[m]]
                q.b[m]['mean'] = q.b[m]['mean'][:,feat_sel[m]]
                q.d[m] = len(feat_sel[m])
        
    def struct_data(self, X, method, sparse = 0, V = None, kernel = None, sig = 'auto', sparse_fs = 0, center = 1, mask = None):
        """Fit model to data.
        
        Parameters
        ----------
        __X: dict, ('data', 'method', 'sparse').
            Dictionary with the information of the input views. Where 'data' 
            stores the matrix with the data. These matrices have size n_samples
            and can have different number of features. If one view has a number
            of samples smaller than the rest, these values are infered assuming
            it is a semisupervised scheme. This dictionary can be built using 
            the function "struct_data".
        __method: char.
            Indicates which type of vraible this is among these:
                'reg'  - regression, floats (shape = [n_samples, n_features]).
                'cat'  - categorical, integers (shape = [n_samples,])
                'mult' - multilabel, one-hot encoding (shape = [n_samples, n_targets])
            
        __sparse: bool, (default 0).
            Indicates if the variable wants to have sparsity in its features 
            or not.
            
        """
        if not (V is None):
            if (kernel is None):
                kernel = 'rbf'
            else:
                kernel = kernel.lower()
            
        X = {"data": X,
        "sparse": sparse,
        "sparse_fs": sparse_fs,
        "method": method,
        "mask": mask,
        "SV": V,
        "kernel": kernel,
        "sig": sig,
        "center": center
        }
        
        if mask is not None and not sparse:
            print('The given mask will not be used as sparsity hasn\'t been selected.')
        if V is not None:
            print('Working on the dual space.')
        return X

    def calcAUC(self, Y_pred, Y_tst):
        n_classes = Y_pred.shape[1]
        
        # Compute ROC curve and ROC area for each class    
        fpr = dict()
        tpr = dict()
        roc_auc = np.zeros((n_classes,1))
        for i in np.arange(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_tst[:,i], Y_pred[:,i]/n_classes)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        return roc_auc.flatten()

    def compute_predictions(self, X_tst, m_in=[0], m_out=1, tr=0):
        
        if None in X_tst:
            X_tst = self.X[m_in[0]]     
        n_tst = self.n_max - self.n[m_out]
        if self.method[m_out] == 'reg':
            if self.SS[m_out]:
                Y_pred = self.X[m_out]['mean'][-n_tst:,:]#.reshape(self.n_max - self.n[m_out], self.d[m_out])
            else:
                [Y_pred, var] = self.predict(m_in, m_out, X_tst)
                
        elif self.method[m_out] == 'cat':
            if self.SS[m_out]:
                Y_pred = self.t[m_out]['mean'][-n_tst:,]
            else:
                Y_pred = self.predict(m_in, m_out, X_tst)
                
        elif self.method[m_out] == 'mult':
            if self.SS[m_out]:
#                Y_pred = self.t[m_out]['mean'][self.SS_mask[m_out]].reshape(self.n_max - self.n[m_out], self.d[m_out])
                if tr == 1:
                    Y_pred = self.q_dist.tS[m_out]['mean'][:self.n[m_out],:]
                else:
                    Y_pred = self.t[m_out]['mean'][-n_tst:,:]
            else:
                Y_pred = self.predict(m_in, m_out, X_tst)
        return Y_pred

    def compute_mse(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        if self.method[m_out] == 'cat':
            Y_pred = np.argmax()
        d = (Y_tst['data'] - Y_pred).ravel()
        return Y_tst['data'].shape[0]**-1 * d.dot(d)
    
    def compute_R2(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        return r2_score(Y_tst['data'], Y_pred[-Y_tst['data'].shape[0]:,:], multioutput = 'uniform_average')
    
    def compute_HL(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        if self.method[m_out] == 'cat':
            Y_pred = label_binarize(Y_pred, classes = np.arange(self.d[m_out]))
            Y_tst_bin = label_binarize(Y_tst['data'], classes = np.arange(self.d[m_out]))  
        elif self.method[m_out] in {'reg', 'mult'}:
            Y_pred = (Y_pred > 0.5).astype(int)
            Y_tst_bin = np.copy(Y_tst['data'])
        return hamming_loss(Y_tst_bin.astype(float), Y_pred)
    
    def compute_AUC(self, Y_tst, X_tst, m_in=[0], m_out=1, tr=0):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out, tr=tr)
        if self.method[m_out] == 'cat':
            # Y_pred = label_binarize(Y_pred, classes = np.arange(self.d[m_out]))
            Y_tst_bin = label_binarize(Y_tst['data'], classes = np.arange(self.d[m_out])) 
        else:
            Y_tst_bin = np.copy(Y_tst['data'])
        if self.method[m_out] in {'reg', 'mult'} and Y_tst['data'].shape[1] != self.d[m_out]:
            Y_pred = np.zeros_like(Y_tst['data']).astype(float)
            Y_pred[:,0] = self.t[1]['mean'][(self.n_max - Y_tst['data'].shape[0]):,:].flatten()
            for i in np.arange(1,Y_tst['data'].shape[1]):
                Y_pred[:,i] = self.t[i+1]['mean'][(self.n_max - Y_tst['data'].shape[0]):,:].flatten()
        p_class = np.sum(Y_tst_bin,axis=0)/np.sum(Y_tst_bin)
        return np.sum(self.calcAUC(Y_pred, Y_tst_bin)*p_class)
    
    def compute_ACC(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        if self.method[m_out] == 'reg':
            Y_pred = np.argmax(Y_pred, axis=1)
            Y_real = np.argmax(Y_tst['data'], axis=1)
        if self.method[m_out] == 'cat':
            Y_real = Y_tst['data'].flatten()
            Y_pred = (np.ones((Y_pred.shape[0], self.d[m_out])) * np.unique(Y_tst['data']))[label_binarize(np.argmax(abs(Y_pred),axis=1), classes = np.arange(self.d[m_out])).astype(bool)]
        if self.method[m_out] == 'mult':
            Y_pred = np.argmax(Y_pred, axis=1)
            if len(Y_tst['data'].shape) < 2:
                Y_real = Y_tst['data'].flatten()
            else:
                Y_real = np.argmax(Y_tst['data'], axis=1)
        return accuracy_score(Y_real, Y_pred)
    
    def update(self, Y_tst=[None], X_tst=[None], X_tr=[None], HL=0, AUC=0, ACC=0, mse=0, R2=0, m_in=[0], m_out=1):
        """Update the variables of the model.
        
        This function updates all the variables of the model and stores the 
        lower bound as well as the Hamming Loss or MSE if specified.
        
        Parameters
        ----------
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
            
        """

        q = self.q_dist   
                
        for m in np.arange(self.m):  
            self.update_w(m)
        self.update_Z()
        for m in np.arange(self.m):   
            #Regression
            if self.method[m] == 'reg':
                if self.SS[m]:
                    # Updating the mean and variance of X2* for the SS case
                    self.update_xs(m)
                    self.X[m]['mean'][self.SS_mask[m]] = q.XS[m]['mean'][self.SS_mask[m]]
                    self.X[m]['cov'][self.SS_mask[m]] = q.XS[m]['cov'][0,0]
                    self.X[m]['prodT'] = np.zeros((q.d[m],q.d[m]))
                    self.X[m]['sumlogdet'] = 0
                    for n in np.arange(self.n_max):                 
                        self.X[m]['prodT'] += np.dot(self.X[m]['mean'][n,np.newaxis].T, self.X[m]['mean'][n,np.newaxis]) + np.diag(self.X[m]['cov'][n,:])
                        self.X[m]['sumlogdet'] += np.linalg.slogdet(np.diag(self.X[m]['cov'][n,self.SS_mask[m][n,:]]))[1]
                else:
                    self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean'])
                #Update of the variable tau
                self.update_tau(m)

            # Categorical
            elif self.method[m] == 'cat': 
                q.tau[m]['a'] = 1
                q.tau[m]['b'] = 1
                self.update_xcat(m)
                if self.SS[m]:
                    self.update_tc(m)
                    self.t[m]['mean'][self.SS_mask[m]] = np.argmax(q.tc[m][self.SS_mask[m]],axis=1)

            # MultiLabel
            elif self.method[m] == 'mult': 
                for i in np.arange(2):
                    self.update_x(m)
                    self.update_xi(m)
                    if self.SS[m]:
                        # Updating the mean and variance of t* for the SS case
                        self.update_t(m)
                        self.t[m]['mean'][self.SS_mask[m]] = q.tS[m]['mean'][self.SS_mask[m]]
                #Update of the variable tau
                self.update_tau(m)
                
            if self.method[m] == 'reg' or self.method[m] == 'mult':
                self.update_b(m)
                    
            if self.sparse[m]:
                self.update_gamma(m)
            self.update_alpha(m)
            
            # if self.sparse_fs[m] and len(self.L) < 50:
            if m in self.k.keys() and self.k[m] == 'rbf':# and len(self.L) < 1000:
                if self.sig[m] == 'auto' or self.sparse_fs[m]:
                    self.sparse_K.sgd_step(q.Z['mean']@q.W[m]['mean'].T, 10)
                    if self.SS[m]:
                        print('Semisupervised version not implemented yet')
                    else:
                        _, kernel, var = self.sparse_K.get_params()
                        self.X[m]['mean'] = self.center_K(kernel)


        if not(None in Y_tst):
            if HL: 
                self.HL.append(self.compute_HL(Y_tst, X_tst, m_in, m_out))
            if AUC:        
                self.AUC.append(self.compute_AUC(Y_tst, X_tst, m_in, m_out))
            if ACC:
                self.ACC.append(self.compute_ACC(Y_tst, X_tst, m_in, m_out))
            if mse:
                self.mse.append(self.compute_mse(Y_tst, X_tst, m_in, m_out))
            if R2:
                self.R2.append(self.compute_R2(Y_tst, X_tst, m_in, m_out))
            
    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:
            L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) #np.linalg.cholesky(A)
            B = np.dot(L.T,L) #linalg.pinv(L)*linalg.pinv(L.T)
            return B
        except:
            return np.nan
        
    def sigmoid(self,x):
        """Computation of the sigmoid function.
        
        Parameters
        ----------
        __x: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        return np.exp(-np.log(1 + np.exp(-x)))
#        return 1. / (1 + np.exp(-x))
  
    def lambda_func(self,x):
        """Computation of the lambda function.
        
        This function calculates the lambda function defined in the paper.
        
        Parameters
        ----------
        __x: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        return np.exp(np.log(self.sigmoid(x) - 0.5) - np.log(2*x))
#        return (self.sigmoid(x) - 0.5)/(2*x)
          
    def update_Z(self):
        """Updates the variables Z.
        
        This function uses the variables of the learnt model to update Z.

        """
        q = self.q_dist
        
        aux = np.eye(q.Kc)
        for m in np.arange(self.m):
            aux += q.tau_mean(m)*q.W[m]['prodT']
        Z_cov = self.myInverse(aux)
        if not np.any(np.isnan(Z_cov)):
            # cov
            q.Z['cov'] = Z_cov
            # mean
            mn = np.zeros((self.n_max,q.Kc))
            for m in np.arange(self.m):
                mn += np.dot(np.subtract(self.X[m]['mean'], q.b[m]['mean']),q.W[m]['mean']) * q.tau_mean(m)
            q.Z['mean'] = np.dot(mn,q.Z['cov'])
            # E[Y*Y^T]
            q.Z['prodT'] = np.dot(q.Z['mean'].T, q.Z['mean']) + self.n_max*q.Z['cov'] 
        else:
            print ('Cov Z is not invertible, not updated')
    
    def update_w(self, m):
        """Updates the variable W.
        
        This function uses the variables of the learnt model to update W of 
        the specified view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        q = self.q_dist
        
        if self.sparse[m]:
            from numpy import sqrt, divide, dot, diag
            q.W[m]['cov'] = np.zeros((q.Kc,q.Kc))
            q.W[m]['prodT'] = np.zeros((q.Kc,q.Kc))
            q.W[m]['prodTalpha'] = np.zeros((q.d[m],))
            q.W[m]['prodTgamma'] = np.zeros((q.Kc,))
            q.W[m]['sumlogdet'] = 0
            
            A = diag(divide(1.,sqrt(q.alpha_mean(m))))
            U, S, UH = np.linalg.svd(dot(dot(A, q.tau_mean(m) * q.Z['prodT']), A), hermitian=True)
            I = diag(dot(UH, U))
            AUH = dot(A, UH.T)
            UA = dot(U.T, A)
            
            if self.area_mask[m] is not None:
                for f in np.arange(len(np.unique(self.area_mask[m]))):
                    # w_cov = self.myInverse(diag(q.alpha_mean(m))*q.gamma_mean(m)[f] + q.tau_mean(m) * q.Z['prodT'])
                    w_cov = dot(AUH * divide(1.,(I*q.gamma_mean(m)[f] + S)), UA)
                    d = self.area_mask[m] == np.unique(self.area_mask[m])[f]
                    q.W[m]['cov'] += w_cov
                    q.W[m]['mean'][d,:] = np.linalg.multi_dot([(self.X[m]['mean'][:,d] - q.b[m]['mean'][0,d]).T, q.Z['mean'] ,w_cov])*q.tau_mean(m)
                    wwT = dot(q.W[m]['mean'][d,:].T, q.W[m]['mean'][d,:]) + w_cov
                    q.W[m]['prodT'] += wwT
                    DwwT = diag(wwT)
                    q.W[m]['prodTgamma'] += q.gamma_mean(m)[f]*DwwT 
                    q.W[m]['prodTalpha'][d] = dot(q.alpha_mean(m),DwwT)
                    q.W[m]['sumlogdet'] += np.linalg.slogdet(w_cov)[1]
            else:
                for d in range(self.d[m]):
                    # w_cov = self.myInverse(diag(q.alpha_mean(m))*q.gamma_mean(m)[d] + q.tau_mean(m) * q.Z['prodT'])
                    w_cov = dot(AUH * divide(1.,(I*q.gamma_mean(m)[d] + S)), UA)
                    if not np.any(np.isnan(w_cov)):
                        q.W[m]['cov'] += w_cov
                        q.W[m]['mean'][d,:] = np.linalg.multi_dot([(self.X[m]['mean'][:,d] - q.b[m]['mean'][0,d]).T, q.Z['mean'] ,w_cov])*q.tau_mean(m)
                        wwT = dot(q.W[m]['mean'][d,np.newaxis].T, q.W[m]['mean'][d,np.newaxis]) + w_cov
                        q.W[m]['prodT'] += wwT
                        DwwT = diag(wwT)
                        q.W[m]['prodTgamma'] += q.gamma_mean(m)[d]*DwwT 
                        q.W[m]['prodTalpha'][d] = dot(q.alpha_mean(m),DwwT)
                        q.W[m]['sumlogdet'] += np.linalg.slogdet(w_cov)[1]
                    else:
                        print ('Cov W is not invertible, not updated')
        else:
            # cov
            # w_cov = self.myInverse(np.diag(q.alpha_mean(m)) + q.tau_mean(m) * q.Z['prodT'])
            # Efficient and robust way of computing:  solve(diag(alpha) + tau * ZZ^T)
            tmp = 1/np.sqrt(q.alpha_mean(m))
            aux = np.outer(tmp,tmp)*q.Z['prodT'] + np.eye(q.Kc)/q.tau_mean(m)
            cho = np.linalg.cholesky(aux)            
            w_cov = 1/q.tau_mean(m) * np.outer(tmp,tmp) * np.dot(linalg.pinv(cho.T),linalg.pinv(cho))
            
            if not np.any(np.isnan(w_cov)):
                q.W[m]['cov'] = w_cov
                # mean
                q.W[m]['mean'] = q.tau_mean(m) * np.linalg.multi_dot([np.subtract(self.X[m]['mean'], q.b[m]['mean']).T,q.Z['mean'],q.W[m]['cov']])
                #E[W*W^T]
                q.W[m]['prodT'] = np.dot(q.W[m]['mean'].T, q.W[m]['mean']) + self.d[m]*q.W[m]['cov']
            else:
                print ('Cov W' + str(m) + ' is not invertible, not updated')           
            
    def update_b(self,m):
        """Updates the variable b.
        
        This function uses the variables of the learnt model to update b of 
        the specified view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        q = self.q_dist
        q.b[m]['cov'] = (1 + self.n_max * q.tau_mean(m))**(-1) * np.eye(self.d[m])
        q.b[m]['mean'] = q.tau_mean(m) * np.dot(np.sum(np.subtract(self.X[m]['mean'], np.dot(q.Z['mean'], q.W[m]['mean'].T)), axis=0)[np.newaxis,:], q.b[m]['cov'])
        q.b[m]['prodT'] = np.sum(q.b[m]['mean']**2) + self.d[m]*q.b[m]['cov'][0,0]    #mean of a noncentral chi-squared distribution
        
    def update_alpha(self,m):
        """Updates the variable alpha.
        
        This function uses the variables of the learnt model to update alpha of 
        the specified view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        q = self.q_dist
        q.alpha[m]['a'] = (self.hyper.alpha_a[m] + 0.5 * self.d[m])/(self.d[m])
        if self.sparse[m]:
            prod = q.W[m]['prodTgamma']
        else:
            prod = np.diag(q.W[m]['prodT'])
        q.alpha[m]['b'] = (self.hyper.alpha_b[m] + 0.5 * prod)/(self.d[m])
        
    def update_tau(self,m):
        """Updates the variable tau.
        
        This function uses the variables of the learnt model to update tau of 
        the specified view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        q = self.q_dist
        q.tau[m]['a'] = (self.hyper.tau_a[m] + 0.5 * self.d[m]*self.n_max)/(self.d[m]*self.n_max) 
        q.tau[m]['b'] = (self.hyper.tau_b[m] + 0.5 *(np.trace(self.X[m]['prodT']) + np.trace(np.dot(q.W[m]['prodT'],q.Z['prodT'])) 
                                                     - 2 * np.trace(np.linalg.multi_dot([q.W[m]['mean'], q.Z['mean'].T,self.X[m]['mean']])) 
                                                     + 2 * np.sum(np.linalg.multi_dot([q.Z['mean'], q.W[m]['mean'].T,q.b[m]['mean'].T])) 
                                                     - 2 *np.sum(np.dot(self.X[m]['mean'],q.b[m]['mean'].T)) 
                                                     + self.n_max * q.b[m]['prodT'] ))/(self.d[m]*self.n_max)

    def update_gamma(self,m):
        """Updates the variable gamma.
        
        This function uses the variables of the learnt model to update gamma of 
        the specified view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        q = self.q_dist     
        q.gamma[m]['a'] = (self.hyper.gamma_a[m] + 0.5 * q.Kc)/q.Kc
        q.gamma[m]['b'] = (self.hyper.gamma_b[m] + 0.5 *q.W[m]['prodTalpha'])/q.Kc
                
    def update_xs(self,m): #Semisupervised
        """Updates the variable X*.
        
        This function uses the variables of the learnt model to update X* of 
        the specified view in the case of semisupervised learning.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        
        q = self.q_dist
        # cov
        q.XS[m]['cov'] = q.tau_mean(m)**(-1)*np.eye(self.d[m])
        # mean
        q.XS[m]['mean'] = np.add(np.dot(q.Z['mean'],q.W[m]['mean'].T), q.b[m]['mean'])
    
    def update_t(self,m): 
        """Updates the variable t*.
        
        This function uses the variables of the learnt model to update t* of 
        the specified view in the case of semisupervised learning.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        
        q = self.q_dist
        # mean
        q.tS[m]['mean'] = self.sigmoid(self.X[m]['mean'])
        # cov
        q.tS[m]['cov'] = np.exp(np.subtract(np.log(q.tS[m]['mean']), np.log((1 + np.exp(self.X[m]['mean'])))))
        # sum(log(det(X)))
        q.tS[m]['sumlogdet'] = 0
        for n in np.arange(self.n[m],self.n_max):
            q.tS[m]['sumlogdet'] += np.linalg.slogdet(np.diag(q.tS[m]['cov'][n,:]))[1]
            
    def update_x(self,m): #Multilabel
        """Updates the variable X.
        
        This function uses the variables of the learnt model to update X of 
        the specified view in the case of a multilabel view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        
        q = self.q_dist
        self.X[m]['prodT'] = np.zeros((q.d[m],q.d[m]))
        self.X[m]['sumlogdet'] = 0
        
        for n in np.arange(self.n_max):
            # cov
            self.X[m]['cov'][n,:] = (q.tau_mean(m) + 2*self.lambda_func(q.xi[m][n,:]))**(-1) #We store only the diagonal of the covariance matrix
            # mean
            self.X[m]['mean'][n,:] = np.dot((self.t[m]['mean'][n,:] - 0.5 + q.tau_mean(m)*(np.dot(q.Z['mean'][n,:],q.W[m]['mean'].T) + q.b[m]['mean'])),np.diag(self.X[m]['cov'][n,:]))
            # prodT
            self.X[m]['prodT'] += np.dot(self.X[m]['mean'][n,np.newaxis].T, self.X[m]['mean'][n,np.newaxis]) + np.diag(self.X[m]['cov'][n,:])
            # sum(log(det(X)))
            self.X[m]['sumlogdet'] += np.linalg.slogdet(np.diag(self.X[m]['cov'][n,:]))[1]

    def update_xi(self,m): #Multilabel    
        """Updates the variable xi.
        
        This function uses the variables of the learnt model to update xi of 
        the specified view in the case of a multilabel view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        q = self.q_dist
        q.xi[m] = np.sqrt(self.X[m]['cov'] + self.X[m]['mean']**2)
        
    def expectation_aprx(self, a, b, c = [None], n_samples = 100, n = None):
        """Calculates the expectation aproximation.
                
        Parameters
        ----------
        __a: float.
            Mean value of the wanted class.
            
        __b: float.
            Mean value of the not wanted classes.
            
        __c: float, (default [None])
            In case there is a pdf in the expectation, this parameter is the one
            used for the mean. N(c - a, 1).
                       
        __n_samples: int, (default 100).
            
        __n: int, (default None).
            

        """

        if n == None:
            n = self.n_max
             
        exp = 0
        for it in np.arange(n_samples):
            u = np.random.normal(0.0, 1.0, n)
            prod = 1
#            prod = 0
            for j in np.arange(np.shape(b)[1]):
                prod = prod * norm.cdf(u + a - b[:,j], 0.0, 1.0) #We calculate the cdf for each class
#                prod = prod + np.log(norm.cdf(u + a - b[:,j], 0.0, 1.0)) #We calculate the cdf for each class
            if not (None in c):
                exp += norm.pdf(u, c - a, 1)*prod
#                exp += np.exp(np.log(norm.pdf(u, c - a, 1)) + prod)
            else:
                exp += prod
#                exp += np.exp(prod)
        return exp/n_samples
    
    def update_xcat(self,m): #Multiclass
        """Updates the variable X.
        
        This function uses the variables of the learnt model to update X of 
        the specified view in the case of a categorical view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        
        q = self.q_dist
        m_worm = np.dot(q.Z['mean'],q.W[m]['mean'].T) + q.b[m]['mean']

        #Obtain the class-wise m_worm
        if self.SS[m]:
            self.X[m]['mean'] = np.zeros((self.n_max,self.d[m]))
            for i in np.arange(self.d[m]-1):
                m_wormi = m_worm[:,i].flatten()
                not_i = np.arange(self.d[m])!=i
                m_wormj = m_worm[:,not_i]
                #Aproximation of the expectation
                xi = self.expectation_aprx(m_wormi, m_wormj)
                #Mean value for Xnj / j!=i
                expj = np.zeros((self.n_max,self.d[m]-1))
                for j in np.arange(self.d[m]-1):
                    m_wormk = m_wormj[:,np.arange(self.d[m]-1)!=j] #it extracts the mean of the values there are neither i nor j
                    expj[:,j] = self.expectation_aprx(m_wormi, m_wormk, c = m_wormj[:,j])
                # mean
                x_i = np.zeros((self.n_max,self.d[m]))
                x_i[:,not_i] = m_wormj - (expj.T/xi).T
                x_i[:,i] = m_wormi + np.sum(m_wormj - x_i[:,not_i],axis=1)
                self.X[m]['mean'] += x_i * q.tc[m][:,i,np.newaxis]
        else:            
            set_classes = np.unique(self.t[m]['data']).astype(int) 
            t_b = label_binarize(self.t[m]['data'], classes=set_classes).astype(bool)
            m_wormi = m_worm[t_b]
            m_wormj = m_worm[~t_b].reshape(self.n_max,self.d[m]-1)
            #Aproximation of the expectation
            xi = self.expectation_aprx(m_wormi, m_wormj) + 1e-10
            #Mean value for Xnj / j!=i
            expj = np.zeros((self.n_max,self.d[m]-1))
            for j in np.arange(self.d[m]-1):
                m_wormk = m_wormj[:,np.arange(self.d[m]-1)!=j] #it extracts the mean of the values there are neither i nor j
                expj[:,j] = self.expectation_aprx(m_wormi, m_wormk, c = m_wormj[:,j])
            # mean
            self.X[m]['mean'][~t_b] = (m_wormj - (expj.T/xi).T).flatten()
            self.X[m]['mean'][t_b] = m_wormi + np.sum(m_wormj - self.X[m]['mean'][~t_b].reshape(self.n_max,self.d[m]-1),axis=1)
            self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean'])
            
    def update_tc(self,m): #Semisupervised categorical

        q = self.q_dist
        m_worm = np.dot(q.Z['mean'],q.W[m]['mean'].T) + q.b[m]['mean']
        for i in np.arange(self.d[m]):
            m_wormi = m_worm[:,np.arange(self.d[m]) == i].flatten()
            m_wormj = m_worm[:,np.arange(self.d[m]) != i]
            q.tc[m][:,i] = self.expectation_aprx(m_wormi, m_wormj)
        # self.X[m]['mean'][~t_b] = (m_wormj - (expj.T/xi).T).flatten()
            
    def predict(self, m_in, m_out, *args):
        """Apply the model learned in the training process to new data.
        
        This function uses the variables of the specified views to predict
        the output view.
        
        Parameters
        ----------
        __X: dict, ('data', 'method', 'sparse').
            Dictionary with the information of the input views. Where 'data' 
            stores the matrix with the data. These matrices have size n_samples
            and can have different number of features. If one view has a number
            of samples smaller than the rest, these values are infered assuming
            it is a semisupervised scheme. This dictionary can be built using 
            the function "struct_data".
            
        __m_in: list. 
            This value indicates which of the views are used as input.        
        __m_out: list. 
            This value indicates which of the input views is used as output.
        """
# =============================================================================
#         Hay que modificarlo para que pueda predecir todo lo que quieras a la vez. 
#         Para ello hay que definir un m_vec = [0,1,0,0,1] indicando qué vistas
#         son para calcular la predicción y cuáles para ser predichas.
# =============================================================================

        q = self.q_dist
        
        if type(args[0]) == dict:
            n_pred = np.shape(args[0]['data'])[0] 
        else:
            n_pred = np.shape(args[0][0]['data'])[0] 
        
        aux = np.eye(q.Kc)
        for m in m_in:
            aux += q.tau_mean(m)*np.dot(q.W[m]['mean'].T,q.W[m]['mean'])
        Z_cov = self.myInverse(aux)
        
        if not np.any(np.isnan(Z_cov)):
            self.Z_mean = np.zeros((n_pred,q.Kc))
            for (m,arg) in enumerate(args):
                if not (arg['SV'] is None) and not(arg['data'].shape[1] == arg['SV'].shape[0]):
                    V = copy.deepcopy(arg['SV'])
                    X = copy.deepcopy(arg['data'])
                    k = copy.deepcopy(arg['kernel'])
                    sig = copy.deepcopy(arg['sig'])
                    center = copy.deepcopy(arg['center'])
                    #Feature selection
                    #Lineal Kernel
                    if k == 'linear':
                        arg['data'] = np.dot(self.X[m]['X'], self.V[m].T)
                    #RBF Kernel
                    elif k == 'rbf': 
                        if sig == 'auto':
                            self.sparse_K = SparseELBO(X, V, self.sparse_fs[m])
                            _, arg['data'], _ = self.sparse_K.get_params()
                        else:
                            arg['data'], sig = self.rbf_kernel_sig(X, V, sig = sig)
                    if center:
                        arg['data'] = self.center_K(arg['data'])
                        
                if type(arg) == dict:
                    if arg['method'] == 'cat': #categorical
                        arg['data'] = label_binarize(arg['data'], classes = np.arange(self.d[m_in[m]]))
                    self.Z_mean += np.dot(arg['data'] - q.b[m]['mean'],q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                else:
                    for (m,x) in enumerate(arg):
                        if x['method'] == 'cat': #categorical
                            x['data'] = label_binarize(x['data'], classes = np.arange(self.d[m_in[m]]))
                        self.Z_mean += np.dot(x['data'] - q.b[m]['mean'],q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                    
            self.Z_mean = np.dot(self.Z_mean,Z_cov)
        else:
            print ('Cov Z is not invertible')
        
        #Regression
        if self.method[m_out] == 'reg':   
            #Expectation X
            mean_x = np.dot(self.Z_mean,q.W[m_out]['mean'].T) + q.b[m_out]['mean']
            #Variance X
            var_x = q.tau_mean(m_out)**(-1)*np.eye(self.d[m_out]) + np.linalg.multi_dot([q.W[m_out]['mean'], Z_cov, q.W[m_out]['mean'].T])                 
            
            return mean_x, var_x
        
        #Categorical
        elif self.method[m_out] == 'cat': 
#            p_t = np.dot(self.Z_mean,q.W[m_out]['mean'].T)
#            pred = np.argmax(abs(p_t),axis=1)
            
            p_t = np.zeros((n_pred,self.d[m_out]))
            m_worm = np.dot(self.Z_mean,q.W[m_out]['mean'].T) + q.b[m_out]['mean']
            for i in np.arange(self.d[m_out]):
                m_wormi = m_worm[:,np.arange(self.d[m_out]) == i].flatten()
                m_wormj = m_worm[:,np.arange(self.d[m_out]) != i]
                p_t[:,i] = self.expectation_aprx(m_wormi, m_wormj, n = n_pred)
            # pred = np.argmax(abs(p_t),axis=1)
            # return pred    
            return p_t
         
        #Multilabel
        elif self.method[m_out] == 'mult': 
            #Expectation X
            m_x = np.dot(self.Z_mean, q.W[m_out]['mean'].T) + q.b[m_out]['mean']  
            #Variance X
            var_x = q.tau_mean(m_out)**(-1)*np.eye(self.d[m_out]) + np.linalg.multi_dot([q.W[m_out]['mean'], Z_cov, q.W[m_out]['mean'].T])
            
            p_t = np.zeros((n_pred,self.d[m_out]))
            #Probability t
            for d in np.arange(self.d[m_out]):
                p_t[:,d] = self.sigmoid(m_x[:,d]*(1+math.pi/8*var_x[d,d])**(-0.5))
            return p_t
       
    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b + sys.float_info.epsilon)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
        
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H
        
    def update_bound(self):
        """Update the Lower Bound.
        
        Uses the learnt variables of the model to update the lower bound.
        
        """
        
        q = self.q_dist
        
        # Entropy of Z
        q.Z['LH'] = self.HGauss(q.Z['mean'], q.Z['cov'], q.Z['LH'])
        for m in np.arange(self.m):
            # Entropy of W
            if self.sparse[m]:
                q.W[m]['LH'] = 0.5*q.W[m]['sumlogdet']
                q.gamma[m]['LH'] = np.sum(self.HGamma(q.gamma[m]['a'], q.gamma[m]['b']))
            else: 
                q.W[m]['LH'] = self.HGauss(q.W[m]['mean'], q.W[m]['cov'], q.W[m]['LH'])
            # Entropy of b
            q.b[m]['LH'] = self.HGauss(q.b[m]['mean'], q.b[m]['cov'], q.b[m]['LH'])
            # Entropy of alpha and tau
            q.alpha[m]['LH'] = np.sum(self.HGamma(q.alpha[m]['a'], q.alpha[m]['b']))
            q.tau[m]['LH'] = np.sum(self.HGamma(q.tau[m]['a'], q.tau[m]['b']))
            # Entropy of X if multilabel
            if self.method[m] == 'mult':
                self.X[m]['LH'] = self.checkInfinity(0.5*self.X[m]['sumlogdet'], self.X[m]['LH'])
            # Entropies if semisupervised 
            if self.SS[m]:
                if self.method[m] == 'reg':
                    q.XS[m]['LH'] = self.checkInfinity(0.5*self.X[m]['sumlogdet'], q.XS[m]['LH'])
                    # q.XS[m]['LH'] = self.HGauss(q.XS[m]['mean'][self.n[m]:,:], q.XS[m]['cov'], q.XS[m]['LH'])
                if self.method[m] == 'mult':
                    q.tS[m]['LH'] = self.checkInfinity(0.5*q.tS[m]['sumlogdet'], q.tS[m]['LH'])

        # Total entropy
        EntropyQ = q.Z['LH']
        for m in np.arange(self.m):
            EntropyQ += q.W[m]['LH'] + q.b[m]['LH'] + q.alpha[m]['LH']  + q.tau[m]['LH']
            if self.sparse[m]:
                EntropyQ += q.gamma[m]['LH']
            if self.method[m] == 'mult':
                EntropyQ += q.X[m]['LH']
            if self.SS[m]:
                if self.method[m] == 'reg':
                    EntropyQ += q.XS[m]['LH']
                if self.method[m] == 'mult':
                    EntropyQ += q.tS[m]['LH']
                    
        # Calculation of the E[log(p(Theta))]
        q.Z['Elogp'] = -0.5*np.trace(q.Z['prodT'])
        for m in np.arange(self.m):   
            q.b[m]['Elogp'] = -0.5*q.b[m]['prodT']
            q.tau[m]['ElogpXtau'] = -(0.5*self.n_max * self.d[m] + self.hyper.tau_a[m] -1)* np.log(q.tau[m]['b'] + sys.float_info.epsilon)
            if self.method[m] == 'mult': #MultiLabel
                q.tau[m]['ElogpXtau'] += np.sum(np.log(self.sigmoid(q.xi[m])) + self.X[m]['mean'] * self.t[m]['mean'] - 0.5 * (self.X[m]['mean'] + q.xi[m]))
            if self.sparse[m]: #Even though it sais Walp, it also includes the term related to gamma
                q.alpha[m]['ElogpWalp'] = -(0.5* self.d[m] + self.hyper.alpha_a[m] -1)* np.sum(np.log(q.alpha[m]['b'])) -(0.5* q.Kc + self.hyper.gamma_a[m] -1)* np.sum(np.log(q.gamma[m]['b'])) #- self.hyper.gamma_b[m]*np.sum(q.gamma_mean(m))
            else:                    
                q.alpha[m]['ElogpWalp'] = -(0.5* self.d[m] + self.hyper.alpha_a[m] -1)* np.sum(np.log(q.alpha[m]['b']))
        
        # Total E[log(p(Theta))]
        ElogP = q.Z['Elogp']
        for m in np.arange(self.m):
            ElogP += q.tau[m]['ElogpXtau'] + q.alpha[m]['ElogpWalp'] + q.b[m]['Elogp']
        return ElogP - EntropyQ
            
class HyperParameters(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, m):
        self.alpha_a = []
        self.alpha_b = []
        self.gamma_a = []
        self.gamma_b = []
        self.tau_a = []
        self.tau_b = []
        self.xi = []
        for m in np.arange(m): 
            self.alpha_a.append(1e-14)
            self.alpha_b.append(1e-14)
            
            self.tau_a.append(1e-14)
            self.tau_b.append(1e-14)
            
            self.gamma_a.append(2)
            self.gamma_b.append(1)

            



    
    
    
        