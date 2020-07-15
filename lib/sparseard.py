# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
import numpy as np
import pyro.contrib.gp as gp


class SparseELBO(nn.Module):

    def __init__(self, X, V, fs=True, lr=1e-3, verbose=False):
        '''
        This class optimizes the lengthscale of each dimension of the X and V data points
        to give ARD to the system.
        Parameters
        ----------
        X : Numpy Array.
            Data array with shape NxD containing the full data points to train.
        V : Numpy Array.
            Support vector array with shape N'xD (N'<<N) containing the points 
            to be used as support vectors.
        fs : bool, optional.
            Choose if you want to do feature selection or not.
        lr : float, optional
            Learning rate to the Adam optimizer. The default is 1e-3.
        verbose: bool, optional.
            Chose if you want to see optimizer's evolution during training
        Returns
        -------
        None.
        '''
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.loss = []
        self.verbose=verbose

        self.X = torch.from_numpy(X).to(self.device)
        self.V = torch.from_numpy(V).to(self.device)
        if fs:
            is_var = False
            var = 1.0            
            is_leng = True
            leng = self.X.shape[1]/10
        else:
            is_var = True
            var = 0.01        
            is_leng = False
            leng = 1.0
        self.kernel = gp.kernels.RBF(input_dim=self.X.shape[1], variance=torch.tensor(var),
                lengthscale= leng * torch.ones([self.X.shape[1],]).to(self.device),
                active_dims = range(X.shape[1]))
        self.kernel.variance_unconstrained.requires_grad = is_var
        self.kernel.lengthscale_unconstrained.requires_grad = is_leng
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        print("Training using:", self.device)
        print("Training Variance: ",is_var)
        print("Training Lengthscale: ",is_leng)
        
        self.K = self.kernel.forward(self.X, self.V)
        
        self.to(self.device)


    def forward(self, ZAT):
        '''
        Defines the RBF kernel and calculate the ELBO.
        Parameters
        ----------
        ZAT : numpy array.
            Matrix product of Z@A.T with shape NxN'.

        Returns
        -------
        M : torch tensor.
            ELBO function as a tensor of shape NxN'. Without doing the summation
            on axes N and N'.

        '''
        self.K = self.kernel.forward(self.X, self.V)
        M = -(self.K.pow(2) - 2 * self.K * ZAT)/2
        return M
    
    def get_params(self):
        '''
        Returns the lengthscale and the variance of the RBF kernel in numpy form

        Returns
        -------
        float32 numpy array
            A row vector of length D with the value of each lengthscale in float32.
        float32
            A float value meaning the variance of the RBF kernel in float32.

        '''
        lengthscale = np.exp(self.kernel.lengthscale_unconstrained.data.cpu().numpy())
        return lengthscale, self.K.data.cpu().numpy(), self.kernel.variance.data.cpu().numpy()
        
        
    def sgd_step(self, ZAT, it):
        '''
        Computes "it" steps of the Adam optimizer to optimize our ELBO.
        Parameters
        ----------
        ZAT : numpy array
            Matrix product of Z@A.T with shape NxN'..
        it : integer
            Integer that indicates how many steps of the optimizer has to do.

        Returns
        -------
        None.

        '''

        ZAT = torch.from_numpy(ZAT).to(self.device)
        for i in range(it):
            self.opt.zero_grad()
            self.ELBO_loss = torch.sum(self.forward(ZAT))
            if self.verbose:
                if i%100==0:
                    print("Loss: ",-self.ELBO_loss)
                    print("Iterations: ",i)
            self.ELBO_loss.backward()
            self.loss.append(-self.ELBO_loss)
            self.opt.step()










        
        
    
        




