import numpy as np
from scipy import linalg
import copy
import sys

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
    >>> import cca
    >>> model = cca.sshiba_methods.SSHIBA(5, 0)
    >>> X0 = myModel_ml.struct_data(X, 0, 0)
    >>> X1 = myModel_ml.struct_data(Y, 0, 0)
    >>> myModel.fit(X0, X1, max_iter = 100)
    >>> prediction = myModel.predict([0], 1, 0, X0_2)
    
    """
    
    def __init__(self, Kc = 2,  hyper = None, X_init = None, 
                 Z_init = None, W_init = [None], alpha_init = None, 
                 tau_init = None, gamma_init = None):
        self.Kc = int(Kc) # Number of  latent variables
        self.hyper = hyper 
        self.X_init = X_init
        self.Z_init = Z_init
        self.W_init = W_init
        self.alpha_init = alpha_init
        self.tau_init = tau_init

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
            
        """
        
        
        n = []
        for (m,arg) in enumerate(args):
            n.append(int(arg['data'].shape[0]))
        self.n_max = np.max(n)
        
        self.n = []
        self.d = []
        self.X = []
        self.method = []
        m = -1 
        for arg in args:
            m += 1
            if not (None in arg['data']):
                self.n.append(int(arg['data'].shape[0]))
                self.d.append(int(arg['data'].shape[1]))
            elif not (None in self.W_init):
                self.n.append(0)
                self.d.append(self.W_init[m]['mean'].shape[0])  
            self.method.append(arg['method'])
            
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
            #Regression
            self.X[m]['data'] = arg['data']
            self.X[m]['mean'] = np.copy(self.X[m]['data'])
            self.X[m]['cov'] = np.zeros((self.X[m]['mean'].shape))

        self.L = []
        self.mse = []
        self.HL = []
        self.AUC = []
        self.ACC = []
        
        self.m = m+1
        if self.hyper == None:
            self.hyper = HyperParameters(self.m)
            
        if not (None in self.W_init):
            self.Kc = self.W_init[0]['mean'].shape[1]
 
        self.q_dist = Qdistribution(self.X, self.n, self.n_max, self.d, self.Kc, self.m, self.method, self.hyper, 
                                    Z_init = self.Z_init, W_init = self.W_init, alpha_init = self.alpha_init, 
                                    tau_init = self.tau_init, gamma_init = self.gamma_init)               
        self.fit_iterate(**kwargs)
    
    def fit_iterate(self, max_iter = 500, tol = 1e-6, verbose = 0):
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
        
        if verbose:
            print ('L(Q) inicial:%.1f' %  (self.update_bound()))
#            print ('HL inicial:  %.4f' %  (self.compute_HL(Y_tst, X_tst)))  
            
        q = self.q_dist
        for i in range(max_iter):
            # Update the variables of the model
            self.update(verbose)
            
            self.L.append(self.update_bound())   
            if verbose:
                print('Iteration %d' %(len(self.L)))
            else:
                print('\rIteration %d Lower Bound %.1f K %4d' %(len(self.L),self.L[-1], q.Kc), end='\r', flush=True)
            # Lower Bound convergence criteria
            if (len(self.L) > 2) and (abs(1 - self.L[-2]/self.L[-1]) < tol):
                print('\nModel correctly trained. Convergence achieved')             
                return
        print('\n')

    def struct_data(self, X, method, sparse = 0, mask = None):
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
        __method: int.
            Indicates which type of vraible this is among these:
                0 - regression, floats (shape = [n_samples, n_features]).
                1 - categorical, integers (shape = [n_samples,])
                2 - multilabel, one-hot encoding (shape = [n_samples, n_targets])
            
        __sparse: bool, (default 0).
            Indicates if the variable wants to have sparsity in its features 
            or not.
            
        """
        
        X = {"data": X,
        "sparse": sparse,
        "method": method,
        "mask": mask
        }
        
        if mask is not None and not sparse:
            print('The given mask will not be used as sparsity hasn\'t been selected.')
        return X

    def update(self, verbose=0):
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

        verboseprint = print if verbose else lambda *a, **k: None
                
        for m in np.arange(self.m):  
            self.update_w(m)
            verboseprint('L(Q) W%i:     %.1f' %  (m+1, self.update_bound()))
        self.update_Z()
        verboseprint('L(Q) Z:      %.1f' %  (self.update_bound()))
        for m in np.arange(self.m):   
            self.update_alpha(m)
            verboseprint('L(Q) alpha%i: %.1f' %  (m+1, self.update_bound()))
            #Regression
            self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean'])
            self.update_tau(m)
            verboseprint('L(Q) tau%i:   %.1f' %  (m+1, self.update_bound()))
            
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
                mn += np.dot(self.X[m]['mean'],q.W[m]['mean']) * q.tau_mean(m)
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

        tmp = 1/np.sqrt(q.alpha_mean(m))
        aux = np.outer(tmp,tmp)*q.Z['prodT'] + np.eye(q.Kc)/q.tau_mean(m)
        cho = np.linalg.cholesky(aux)            
        w_cov = 1/q.tau_mean(m) * np.outer(tmp,tmp) * np.dot(linalg.pinv(cho.T),linalg.pinv(cho))
        
        if not np.any(np.isnan(w_cov)):
            q.W[m]['cov'] = w_cov
            # mean
            q.W[m]['mean'] = q.tau_mean(m) * np.linalg.multi_dot([self.X[m]['mean'].T,q.Z['mean'],q.W[m]['cov']])
            #E[W*W^T]
            q.W[m]['prodT'] = np.dot(q.W[m]['mean'].T, q.W[m]['mean']) + self.d[m]*q.W[m]['cov']
        else:
            print ('Cov W' + str(m) + ' is not invertible, not updated')           
  
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
                                                     - 2 * np.trace(np.linalg.multi_dot([q.W[m]['mean'], q.Z['mean'].T,self.X[m]['mean']]))))/(self.d[m]*self.n_max)
    
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
            Z_mean = np.zeros((n_pred,q.Kc))
            for (m,arg) in enumerate(args):
                Z_mean += np.dot(arg['data'], q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
            Z_mean = np.dot(Z_mean,Z_cov)
        else:
            print ('Cov Z is not invertible')
        
        #Expectation X
        mean_x = np.dot(Z_mean,q.W[m_out]['mean'].T)
        #Variance X
        var_x = q.tau_mean(m_out)**(-1)*np.eye(self.d[m_out]) + np.linalg.multi_dot([q.W[m_out]['mean'], Z_cov, q.W[m_out]['mean'].T])                 
        
        return mean_x, var_x
 
       
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
        __h: float.
            The entropy of the update.        
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
            q.W[m]['LH'] = self.HGauss(q.W[m]['mean'], q.W[m]['cov'], q.W[m]['LH'])
            # Entropy of alpha and tau
            q.alpha[m]['LH'] = np.sum(self.HGamma(q.alpha[m]['a'], q.alpha[m]['b']))
            q.tau[m]['LH'] = np.sum(self.HGamma(q.tau[m]['a'], q.tau[m]['b']))

        # Total entropy
        EntropyQ = q.Z['LH']
        for m in np.arange(self.m):
            EntropyQ += q.W[m]['LH'] + q.alpha[m]['LH']  + q.tau[m]['LH']
                    
        # Calculation of E[log(p(Theta))]
        q.Z['Elogp'] = -0.5*np.trace(q.Z['prodT'])
        for m in np.arange(self.m):   
            q.tau[m]['ElogpXtau'] = -(0.5*self.n_max * self.d[m] + self.hyper.tau_a[m] -1)* np.log(q.tau[m]['b'] + sys.float_info.epsilon)                
            q.alpha[m]['ElogpWalp'] = -(0.5* self.d[m] + self.hyper.alpha_a[m] -1)* np.sum(np.log(q.alpha[m]['b']))
        
        # Total E[log(p(Theta))]
        ElogP = q.Z['Elogp']
        for m in np.arange(self.m):
            ElogP += q.tau[m]['ElogpXtau'] + q.alpha[m]['ElogpWalp']
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
        self.tau_a = []
        self.tau_b = []
        for m in np.arange(m): 
            self.alpha_a.append(1e-14)
            self.alpha_b.append(1e-14)
            
            self.tau_a.append(1e-14)
            self.tau_b.append(1e-14)

class Qdistribution(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, X, n, n_max, d, Kc, m, method, hyper, Z_init=None, 
                 W_init=None, alpha_init=None, tau_init=None, gamma_init=None):
        self.n = n
        self.n_max = n_max
        self.d = d
        self.Kc = Kc
        self.m = m
        self.X = X
        # Initialize some parameters that are constant
        alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.m,(self.Kc*np.ones((self.m,))).astype(int))
        self.alpha = alpha if alpha_init is None else alpha_init
        tau = self.qGamma(hyper.tau_a,hyper.tau_b,self.m,(np.ones((self.m,))).astype(int))
        self.tau = tau if tau_init is None else tau_init
            
        # The remaning parameters at random 
        self.init_rnd(X, method, Z_init, W_init)

    def init_rnd(self, X, method, Z_init=None, W_init=None):
        """ Hyperparameter initialisation.
    
        Parameters
        ----------
        __m : int.
            number of views in the model.
            
        """
        
        W = []
        for m in np.arange(self.m):
            info = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            W.append(info)
        Z = copy.deepcopy(W[0])
            
        # Initialization of the latent space matrix Z
        Z['mean'] = np.random.normal(0.0, 1.0, self.n_max * self.Kc).reshape(self.n_max, self.Kc)
        Z['cov'] = np.eye(self.Kc)
        Z['prodT'] = Z['cov'] + self.n_max*Z['cov']
        if Z_init is None: #If the matrix is not initialised
            self.Z = Z
        elif Z_init['mean'].shape[0]<self.n_max: #If only the supervised part of the matrix is initialised
            self.Z = Z
            self.Z['mean'][:Z_init['mean'].shape[0],:] = Z_init['mean']
            self.Z['cov'] = Z_init['cov']
            self.Z['prodT'] = Z_init['prodT']
        else: #If all the matrix is initialised          
            self.Z = Z_init
        
        for m in np.arange(self.m):
            # Initialization of the matrix W for each view
            W[m]['mean'] = np.random.normal(np.zeros((self.d[m],self.Kc)), 1/(np.repeat(self.alpha_mean(m).reshape(1,self.Kc),self.d[m],axis=0))) #np.random.normal(0.0, 1.0, self.d[m] * self.Kc).reshape(self.d[m], self.Kc)
            W[m]['cov'] = np.dot(W[m]['mean'].T,W[m]['mean']) #np.eye(self.Kc)
            W[m]['prodT'] = np.dot(W[m]['mean'].T, W[m]['mean'])+self.Kc*W[m]['cov']

        self.W = W if None in W_init else W_init
               
    def qGamma(self,a,b,m_i,r):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [m_in, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [m_in, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __r: array (shape = [m_in, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = []
        for m in np.arange(m_i): 
            info = {                
                "a":         a[m],
                "LH":         None,
                "ElogpWalp":  None,
            }
            info["b"] = (b[m]*np.ones((r[m],1))).flatten()
            param.append(info)
        return param
    
    def alpha_mean(self,m):
        """ Mean of alpha.
        It returns the mean value of the variable alpha for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.alpha[m]["a"] / self.alpha[m]["b"]
    
    def tau_mean(self,m):
        """ Mean of tau.
        It returns the mean value of the variable tau for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.tau[m]["a"] / self.tau[m]["b"]

    def gamma_mean(self,m):
        """ Mean of gamma.
        It returns the mean value of the variable gamma for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.gamma[m]["a"] / self.gamma[m]["b"]