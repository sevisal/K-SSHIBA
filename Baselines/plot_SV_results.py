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


stack_dict = {}
stack_dict['atp1d'] =  []
stack_dict['atp7d'] =  []
stack_dict['oes97'] =  []
stack_dict['oes10'] =  []
stack_dict['edm'] =    []
stack_dict['jura'] =   []
stack_dict['wq'] =     []
stack_dict['enb'] =    []




v_dim = np.arange(0.05, 1.05, 0.05)
# =============================================================================
# KPCA FIGURAS
show = "KPCA_LR"
file = "_kpca_from1to100.pkl"
file2 = "--"

# =============================================================================

#KCCA FIGURAS
# show = "KCCA_LR"
# file = "_kcca_test.pkl"
# file2 = "--"


results_r2 = {}
results_lf = {}
SV_opt = {}
for key in my_dict.keys():
    opt_name = key+"_opt_value.pkl"
    stacked = False
    filename = key+file
    filename2 = key+file2
    if os.path.exists(filename):
        print("------------------------------------------------")
        print("Database trained: "+key)
        my_dict[key].append(pickle.load( open( filename, "rb" )))
    if os.path.exists(filename2):
        stacked = True
        print("This database has been trained more than once")
        stack_dict[key].append(pickle.load( open( filename2, "rb" )))
    if os.path.exists(opt_name):
        opt_value = pickle.load( open( opt_name, "rb" ))
    if len(my_dict[key]):
        r2_mean = {}
        r2_std = {}
        lf_mean = {}
        lf_std = {}
        sv_mean = {}
        opt_mean = {}
        opt_std = {}
        
        
        for base in my_dict[key][0][0].keys():
            

            print("---------- Results of: "+str(base)+"-------------")
            print("R2 training")
            
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
                sv_mean[base] = np.argmax(np.mean(my_dict[key][0][0][base][0], axis=2), axis = 1)
                
            opt_mean[base] = np.mean(opt_value[base]['R2'])
            opt_std[base] = np.std(opt_value[base]['R2'])
            
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
            
            if base=="KPCA_LR":
                print("LF analysis at optimum value: %5.0f +/- %5.0f" %(np.mean(opt_value[base]['Kc']), np.std(opt_value[base]['Kc'])))
                

            print("Optimum R2: %0.3f +/- %0.3f" %(opt_mean[base], opt_std[base])) 
            print("Using %3.2f +/- %3.2f of support vectors" %(np.mean(v_dim[sv_mean[base]])*100, 100*np.std(v_dim[sv_mean[base]])))
            print("R2 using 100% of sv: "+ str(r2_mean[base][-1]))
            print("STD using 100% of sv: "+ str(r2_std[base][-1]))
            plt.figure()
            plt.plot(v_dim, r2_mean[base])
            plt.plot(v_dim[index_max_r2], r2_mean[base][index_max_r2] , 'r*')
            plt.xlabel("% of support vectors used")
            plt.ylabel("R2 score")
            plt.title(key+"_"+base)
            plt.show()
        SV_opt[key] = sv_mean
        results_r2[key] = r2_mean
        results_lf[key] = lf_mean
    
pickle.dump(SV_opt, open('sv_opt_kcca.pkl', "wb" ))
    

        
        
            
            
             