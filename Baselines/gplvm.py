# -*- coding: utf-8 -*-

import GPy
from scipy.io import arff
import pandas as pd
import pickle
import numpy as np

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
my_dict['andro'] =  [6]
my_dict['osales'] = [12]
my_dict['scpf'] =   [3]

database = "atp7d"

with open('Paper_results.pkl', 'wb') as output:
    pickle.dump(my_dict, output, pickle.HIGHEST_PROTOCOL)
    
paper_res = pickle.load( open( 'Paper_results.pkl', "rb" ), encoding='latin1' )


data, meta = arff.loadarff('../Databases/MultiTaskRegressionDatasets/' + database +'.arff')

df = pd.DataFrame(data)
n_classes = paper_res[database][0]
X = np.array(df)[:,:-n_classes].astype(float)
Y = np.array(df)[:,-n_classes:].astype(float)
idx = np.random.randint(0,2,Y.shape[0]).astype(int)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


m = GPy.models.GPLVM(X, int(X.shape[1]/2))
m.optimize('bfgs', messages=1, max_f_eval=10000)

