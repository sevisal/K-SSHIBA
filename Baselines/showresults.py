# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

paper_res = pickle.load( open( 'Paper_results.pkl', "rb" ), encoding='latin1' )

for database in paper_res:
    print("-----------------------DATABASE "+database+"----------------------")
    folds = 10
    folds_file = str(folds)+'folds_'+database+'.p'
    if os.path.exists(folds_file):
        print('Loading test and validation folds.')
        [fold_tst, dict_fold_val] = pickle.load(open(folds_file, 'rb'))
    for i in np.arange(len(fold_tst)):
        # At his point we check whether the file where we want to store the results does or doesn't already exist.
        # If it does we check if this baseline has been stored and, if so, we load it. If the baseline isn't in the file, we define it.
        filename = 'Results/Baselines_'+database+'_'+str(folds)+'folds.pkl'
        if os.path.exists(filename):
            results = pickle.load( open( filename, "rb" ) )

    bases= ['_NN']
    for base in bases:
        print(base +' mean R2:  %0.3f +/- %0.3f%%' %(np.mean(results[base]['R2']) , np.std(results[base]['R2'])))
        print(base +' mean MSE: %0.3f +/- %0.3f' %(np.mean(results[base]['mse']) , np.std(results[base]['mse'])))
        print(base +' mean Kc:  %0.3f +/- %0.3f' %(np.mean(results[base]['Kc']) , np.std(results[base]['Kc'])))
