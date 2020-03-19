# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

paper_res = pickle.load( open( 'Paper_results.pkl', "rb" ), encoding='latin1' )
r2_scores = np.zeros((len(paper_res),4,2))
mse_scores = np.zeros((len(paper_res),4,2))
kcs = np.zeros((len(paper_res),4,2))
for e,database in enumerate(paper_res):
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
            
    if len(results) == 4:
        bases= ['KPCA_LR', 'KCCA_', 'KCCA_LR']
    else:
        bases= ['KPCA_LR', 'KCCA_', 'KCCA_LR']
    for j,base in enumerate(bases):
        r2_scores[e,j,0] = np.mean(results[base]['R2'])
        r2_scores[e,j,1] = np.std(results[base]['R2'])
        print(base +' mean R2:  %0.3f +/- %0.3f%%' %(np.mean(results[base]['R2']) , np.std(results[base]['R2'])))
        mse_scores[e,j,0] = np.mean(results[base]['mse'])
        mse_scores[e,j,1] = np.std(results[base]['mse'])
        print(base +' mean MSE: %0.3f +/- %0.3f' %(np.mean(results[base]['mse']) , np.std(results[base]['mse'])))
        kcs[e,j,0] = np.mean(results[base]['Kc'])
        kcs[e,j,1] = np.std(results[base]['Kc'])
        print(base +' mean Kc:  %0.3f +/- %0.3f' %(np.mean(results[base]['Kc']) , np.std(results[base]['Kc'])))

import seaborn as sns
import matplotlib.pyplot as plt
bases= ['KPCA_LR', 'KCCA_', 'KCCA_LR']
databases= list(paper_res.keys())
plt.figure()
ax = sns.heatmap(r2_scores[:,:,0], annot=True, xticklabels=bases, yticklabels=databases)
ax.set_title('R2 scores for each database')
plt.show()

plt.figure()
ax2 = sns.heatmap(mse_scores[:,:,0], annot=True, xticklabels=bases, yticklabels=databases)
ax2.set_title('MSE scores for each database')
plt.show()

plt.figure()
ax2 = sns.heatmap(kcs[:,:,0], annot=True, xticklabels=bases, yticklabels=databases)
ax2.set_title('Kc for each database')
plt.show()
