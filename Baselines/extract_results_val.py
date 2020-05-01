# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

my_dict = {}
my_dict['atp1d'] =  []
my_dict['atp7d'] =  []
my_dict['oes97'] =  []
my_dict['oes10'] =  []
my_dict['edm'] =    []
my_dict['jura'] =   []
my_dict['wq'] =     []
my_dict['enb'] =    []
my_dict['slump'] =  []

stack_dict = {}
stack_dict['atp1d'] =  []
stack_dict['atp7d'] =  []
stack_dict['oes97'] =  []
stack_dict['oes10'] =  []
stack_dict['edm'] =    []
stack_dict['jura'] =   []
stack_dict['wq'] =     []
stack_dict['enb'] =    []
stack_dict['slump'] =  []


def r2_on_validation():
    results = {}    
    for key in my_dict.keys():
        filename = key+"_r2_ss_tf_vf_5.pkl"
        if os.path.exists(filename):
            print("------------------------------------------------")
            print("Database trained: "+key)
            my_dict[key].append(pickle.load( open( filename, "rb" )))
          
        if len(my_dict[key]): 
            random_val_mean = {}
            for base in my_dict[key][0].keys():
                print("Baseline trained: " +base)
                random_val_mean[base] = np.mean(my_dict[key][0][base][0], axis=(1,2,3))
                index_max_r2 = np.argmax(random_val_mean[base])
                print("R2 max: "+str(random_val_mean[base][index_max_r2]))
                print("Using "+str(range(5,105,5)[index_max_r2])+" % of support vectors.")
                plt.figure()
                plt.plot(range(5,105,5), random_val_mean[base])
                plt.xlabel("R2 score")
                plt.ylabel("% of support vectors used")
                plt.title(key+"_"+base)
                plt.show()
            results[key] = random_val_mean
            
    return results


results_r2 = {}
results_lf = {}
for key in my_dict.keys():
    stacked = False
    filename = key+"_kcca_test.pkl"
    filename2 = key+"_kcca_tostack_test.pkl"
    if os.path.exists(filename):
        print("------------------------------------------------")
        print("Database trained: "+key)
        my_dict[key].append(pickle.load( open( filename, "rb" )))
    if os.path.exists(filename2):
        stacked = True
        print("This database has been trained more than once")
        stack_dict[key].append(pickle.load( open( filename2, "rb" )))
        
    if len(my_dict[key]):
        r2_mean = {}
        lf_mean = {}
        for base in my_dict[key][0][0].keys():
            if base == 'KCCA_LR':
                print("R2 training")
                print("Baseline trained: " +base)
                if stacked:
                    randomness = my_dict[key][0][0][base][0].shape[2]+stack_dict[key][0][0][base][0].shape[2]
                    print("Randomness of: "+str(randomness))
                    aux = np.mean(my_dict[key][0][0][base][0], axis=(0,2))
                    aux2 = np.mean(stack_dict[key][0][0][base][0], axis=(0,2))
                    r2_mean[base] = np.mean(np.stack((aux, aux2)), axis=0)
                else:
                    randomness = my_dict[key][0][0][base][0].shape[2]
                    print("Randomness of: "+str(randomness))
                    r2_mean[base] = np.mean(my_dict[key][0][0][base][0], axis=(0,2))
                index_max_r2 = np.argmax(r2_mean[base])
    
                print("Latent Factor Analysis")
                if stacked:
                    aux = np.mean(my_dict[key][0][1][base][0], axis=(0,2))
                    aux2 = np.mean(stack_dict[key][0][1][base][0], axis=(0,2))
                    lf_mean[base] = np.mean(np.stack((aux, aux2)), axis=0)
                else:
                    lf_mean[base] = np.mean(my_dict[key][0][1][base][0], axis=(0,2))
                print("Latent Factors used that gives the max: "+str(lf_mean[base][index_max_r2]))
                print("R2 max: "+str(r2_mean[base][index_max_r2]))
                print("Using "+str(range(5,105,5)[index_max_r2])+" % of support vectors.")
                plt.figure()
                plt.plot(range(5,105,5), r2_mean[base])
                plt.plot(range(5,105,5)[index_max_r2], r2_mean[base][index_max_r2] , 'r*')
                plt.xlabel("R2 score")
                plt.ylabel("% of support vectors used")
                plt.title(key+"_"+base)
                plt.show()
            else:
                print("KPCA_LR is not done yet.")
        results_r2[key] = r2_mean
        results_lf[key] = lf_mean
        
    

        
        
            
            
             