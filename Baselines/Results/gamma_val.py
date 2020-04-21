# -*- coding: utf-8 -*-

if val:
                    gamma_val = np.array([1./8, 1./4, 1./2, 1, 2, 4, 8])/(np.sqrt(n_classes))
                    r2_val = np.zeros((len(fold_tst),len(gamma_val)))
                    for j in np.arange(len(fold_tst)):
                        pos_tr2 = dict_fold_val[i][j][0]
                        pos_val =  dict_fold_val[i][j][1]
                        Y_val = Y_tr[pos_val]
                        Y_tr2 = Y_tr[pos_tr2]
                        X_val = X_tr[pos_val,:]
                        X_tr2 = X_tr[pos_tr2,:]
                        
                        scaler = StandardScaler()
                        X_tr2 = scaler.fit_transform(X_tr2)
                        X_val = scaler.transform(X_val)
                        
                        for p, g in enumerate(gamma_val):
                            K_tr, sig = rbf_kernel_sig(X_tr2, X_tr2, sig=np.sqrt(1/(2*g)))
                            K_val, sig = rbf_kernel_sig(X_val, X_tr2, sig=sig)
                            
                            K_tr = center_K(K_tr)
                            K_val = center_K(K_val)
                            print(j)
                            print(p)
                            
                            if pipeline[0] == 'KPCA':
                                pca = PCA()
                                P_tr = pca.fit_transform(K_tr)
                                P_tst = pca.transform(K_val)
                                # Selecting the latent factors that explain 95% of the variance.
                                Kc = 0
                                while np.sum(pca.explained_variance_ratio_[:Kc]) < 0.95:
                                    Kc = Kc + 1
                                P_tr = P_tr[:, :Kc]
                                P_tst = P_tst[:, :Kc]
                                
                            elif pipeline[0] == 'KCCA':
                                # KCCA
                                cca = CCA(n_components = Y_tr.shape[1]-1).fit(K_tr, Y_tr2)
                                P_tr = cca.transform(K_tr)
                                P_tst = cca.transform(K_val)
                            if pipeline[1] == 'LR':
                                # Linear Regression
                                reg = LinearRegression()
                                reg.fit(P_tr, Y_tr2)
                                Y_pred = reg.predict(P_tst)
                                r2_val[j,p] = r2_score(Y_val, Y_pred, multioutput = 'uniform_average')
                                
                    r2_mean = np.mean(r2_val, axis=0)
                    gamma = gamma_val(np.argmax(r2_mean))
                    sig = np.sqrt(1/(2*gamma))
                else:
                    sig = 0