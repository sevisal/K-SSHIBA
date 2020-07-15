import numpy as np
import copy

class Qdistribution(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, X, n, n_max, d, Kc, m, sparse, method, SS, SS_mask, area_mask, hyper, Z_init=None, 
                 W_init=None, alpha_init=None, tau_init=None, gamma_init=None):
        self.n = n
        self.n_max = n_max
        self.d = d
        self.Kc = Kc
        self.m = m
        self.sparse = sparse
        self.SS = SS
        self.X = X
        # Initialize some parameters that are constant
        alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.m,(self.Kc*np.ones((self.m,))).astype(int))
        self.alpha = alpha if alpha_init is None else alpha_init
        tau = self.qGamma(hyper.tau_a,hyper.tau_b,self.m,(np.ones((self.m,))).astype(int))
        self.tau = tau if tau_init is None else tau_init
        # We generate gamma for all views, although the ones we are going to use 
        # and update are the ones for which the sparsity has been specified
        gamma = self.qGamma(hyper.gamma_a,hyper.gamma_b,self.m,self.d, area_mask)
        self.gamma = gamma if gamma_init is None else gamma_init
        
        self.xi = []
        for m in np.arange(self.m):            
            self.xi.append(np.sqrt(self.X[m]['cov'] + self.X[m]['mean']**2))
            
        # The remaning parameters at random 
        self.init_rnd(X, method, SS, SS_mask, Z_init, W_init)

    def init_rnd(self, X, method, SS, SS_mask, Z_init=None, W_init=None):
        """ Hyperparameter initialisation.
    
        Parameters
        ----------
        __m : int.
            number of views in the model.
            
        """

        W = []
        self.XS = []
        for m in np.arange(self.m):
            info = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            W.append(info)
        self.XS = copy.deepcopy(W)
        self.tS = copy.deepcopy(W)
        self.b = copy.deepcopy(W)
        self.tc = {}
        Z = copy.deepcopy(W[0])
            
        # Initialization of the latent space matrix Z
        Z['mean'] = np.random.normal(0.0, 1.0, (self.n_max, self.Kc))
        Z['cov'] = np.eye(self.Kc) #np.dot(self.Z['mean'].T,self.Z['mean']) 
        Z['prodT'] = Z['cov'] + self.n_max*Z['cov'] #np.dot(self.Z['mean'].T, self.Z['mean']) + self.n_max*self.Z['cov']
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
            # Initialization of the unknown data
            if self.SS[m]:
                self.tc[m] = np.random.rand(self.n_max, self.d[m])
                self.tS[m]['mean'] = np.random.randint(2, size=[self.n_max, self.d[m]])
                self.tS[m]['cov'] = np.eye(self.d[m]) 
                self.tS[m]['sumlogdet'] = 0
                self.XS[m]['mean'] = np.random.normal(0.0, 1.0, (self.n_max, self.d[m]))
                # self.XS[m]['cov'] = np.eye(self.d[m]) 
                self.XS[m]['cov'] = np.zeros(self.XS[m]['mean'].shape)


            # Initialization of the matrix W for each view
            #np.random.normal(np.zeros((self.d[m],self.Kc)), 1/(np.repeat(self.alpha_mean(m).reshape(1,self.Kc),self.d[m],axis=0)))
            W[m]['mean'] = np.random.normal(0.0, 1.0, (self.d[m], self.Kc))
            W[m]['cov'] = np.dot(W[m]['mean'].T,W[m]['mean']) #np.eye(self.Kc)
            W[m]['prodT'] = np.dot(W[m]['mean'].T, W[m]['mean'])+self.Kc*W[m]['cov']
            if self.sparse[m]:
                W[m]['prodTalpha'] = np.zeros((self.d[m],))
                W[m]['prodTgamma'] = np.zeros((self.Kc,))
                W[m]['sumlogdet'] = 0
            
            if method[m] == 'reg' or method[m] == 'mult':
                self.b[m]['cov'] = (1 + self.n[m] * self.tau_mean(m))**(-1) * np.eye(self.d[m])
                self.b[m]['mean'] = self.tau_mean(m) * np.dot(np.sum(self.X[m]['mean'] - np.dot(self.Z['mean'], W[m]['mean'].T), axis=0)[np.newaxis,:], self.b[m]['cov'])
                self.b[m]['prodT'] = np.sum(self.b[m]['mean']**2) + self.d[m]*self.b[m]['cov'][0,0]    #mean of a noncentral chi-squared distribution
            else:
                self.b[m]['cov'] = np.zeros((self.d[m],self.d[m]))
                self.b[m]['mean'] = np.zeros((self.d[m],))
                self.b[m]['prodT'] = np.sum(np.square(self.b[m]['mean'])) + self.d[m]*self.b[m]['cov'][0,0]    #mean of a noncentral chi-squared distribution

        self.W = W if None in W_init else W_init
               
    def qGamma(self,a,b,m_i,r,mask=None):
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
            if mask is None or mask[m] is None:
                info["b"] = (b[m]*np.ones((r[m],1))).flatten()
            else:
                info["b"] = (b[m]*np.ones((len(np.unique(mask[m])),1))).flatten()
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