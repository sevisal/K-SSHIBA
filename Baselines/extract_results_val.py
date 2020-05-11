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


# stack = np.arange(0.05,0.55,0.05)
# v_dim = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04]
# for st in stack:
#     v_dim.append(st)
    
v_dim = np.arange(0.05, 1.05, 0.05)

results_r2 = {}
results_lf = {}
for key in my_dict.keys():
    stacked = False
    filename = key+"_kpca_from1to100.pkl"
    filename2 = key+"_kpca_tostack_test.pkl"
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
        r2_std = {}
        lf_mean = {}
        lf_std = {}
        for base in my_dict[key][0][0].keys():
            if base == 'KPCA_LR':
                print("----KPCA_LR---")
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
                
                r2_std[base] = np.std(my_dict[key][0][0][base][0], axis=(0,2))
                
                index_max_r2 = np.argmax(r2_mean[base])
    
                print("Latent Factor Analysis")
                if stacked:
                    aux = np.mean(my_dict[key][0][1][base][0], axis=(0,2))
                    aux2 = np.mean(stack_dict[key][0][1][base][0], axis=(0,2))
                    lf_mean[base] = np.mean(np.stack((aux, aux2)), axis=0)
                else:
                    lf_mean[base] = np.mean(my_dict[key][0][1][base][0], axis=(0,2))
                lf_std[base] = np.std(my_dict[key][0][1][base][0], axis=(0,2))
                print("Latent Factors used that gives the max: "+str(lf_mean[base][index_max_r2]))
                print("Latent factors std: "+str(lf_std[base][index_max_r2]))
                print("R2 max: "+str(r2_mean[base][index_max_r2]))
                print("Std: "+str(r2_std[base][index_max_r2]))
                print("Using "+str(v_dim[index_max_r2]*100)+" % of support vectors.")
                print("R2 using 100% of sv: "+ str(r2_mean[base][-1]))
                print("STD using 100% of sv: "+ str(r2_std[base][-1]))
                plt.figure()
                plt.plot(v_dim, r2_mean[base])
                plt.plot(v_dim[index_max_r2], r2_mean[base][index_max_r2] , 'r*')
                plt.xlabel("% of support vectors used")
                plt.ylabel("R2 score")
                plt.title(key+"_"+base)
                plt.show()

        results_r2[key] = r2_mean
        results_lf[key] = lf_mean
        
    

        
        
            
            
             