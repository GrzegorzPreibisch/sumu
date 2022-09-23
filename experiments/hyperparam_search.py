# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sumu
import time
import random
import json
from sklearn import preprocessing

from glmnet import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def dag_to_mat(dag, p, est=True):
    mat = np.zeros((p,p))
    for i in range(p):
       if i in dag:
         for x in dag[i]:
            if est:
                mat[i,x[0]] = 1
            else:
                mat[i,x] =1
    return mat


def dag_to_mat_true(arcs, colnames):
    p = len(colnames)
    mat = np.zeros((p,p))
    for arc in arcs:
        mat[arc[1], arc[0]] = 1
    return mat

def compute_stats(dag_est, dag_true, p):
    mat_t = dag_true
    mat_e = dag_to_mat(dag_est,p,True)
    TP =  np.sum(mat_t*mat_e)
    WD =  np.sum(mat_t*mat_e.transpose())
    FP =  -np.sum((mat_t-1)*mat_e)
    TN =  np.sum((mat_t-1)*(mat_e-1))
    FN =   -np.sum((mat_t)*(mat_e-1))
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    diff = np.abs(mat_t - mat_e)
    diff = diff + diff.transpose()
    diff[diff > 1] = 1  # Ignoring the double edges.
    SHD = np.sum(diff)/2
    return TPR,FPR,SHD, TP,FP,TN,FN,WD, np.sum(mat_t),np.sum(mat_e)

def exp_decay(penalty,step):
        return penalty*step

def division_decay(penalty,step, layer, previous_parent_len, parent_len, normalizing_factor):
    new_penalty =  np.log(previous_parent_len)*normalizing_factor
    print(new_penalty)
    return new_penalty

def division_decay_bic(penalty,step, layer, previous_parent_len, parent_len, normalizing_factor):
    return penaltya

def experiment(array, num_var,dag_true, sizes, b_start_coef_list, g_start_coef_list, 
decay_bic_list, decay_gic_list , penalty_bic_decay_pattern,
 penalty_gic_decay_pattern,filename, N, recurring = False):
    result = []
    statistics = []
    for i in range(N):
        for size in sizes:
            stats = np.zeros(10)
            np.random.shuffle(arr)
            sample = arr[0:size]
            np.savetxt(f"../results/sample{i}_arth150_1000.csv",sample,delimiter=";")
            # sample
            # warstwy
            data = sumu.Data(sample)
            
            g =  sumu.Gadget(data=data, recurring=recurring)
            g.sample()
            for b_start_coef in b_start_coef_list:
                for g_start_coef in g_start_coef_list:
                    for decay_bic in decay_bic_list:
                        for decay_gic in decay_bic_list:
                            b_penalty = np.log(size)*b_start_coef
                            g_penalty = np.log(num_var)*g_start_coef
                            dag_est1, intercept, layers = g.generate_final_dag(pen_bic= b_penalty, pen_gic=g_penalty,
                            step_bic = decay_bic, step_gic = decay_gic,
                                penalty_bic_decay_pattern = penalty_bic_decay_pattern, penalty_gic_decay_pattern = penalty_gic_decay_pattern, normalizing_factor = g_start_coef)
                            layers_list = [[float(y+1) for y in x] for x in layers]
                            with open(f"../results/arth150_layers{i}_1000.json", 'w') as fp:
                            	json.dump(layers_list, fp) 
                            np.savetxt(f"../results/arth150_dag_est{i}_1000.csv",dag_to_mat(dag_est1,num_var),delimiter=";")   
                            comp = compute_stats(dag_est1, dag_true, num_var)
                            print(comp)
                            #statistics.append([i,size, b_start_coef, g_start_coef, decay_bic,decay_gic , comp[0], comp[1],comp[2]])
    #df = pd.DataFrame(statistics, columns = ['Experiment_number','size', 'b_start_coef', 'g_start_coef', 'decay_bic','decay_gic','TPR','FPR','SHD' ])
    #df.to_csv(filename)
    #print(df) 
    statistics =[]
    results = [] 
    return statistics, result, layers, dag_est1, sample

sizes = [1000]

############
############  ecoli70
print('arth150')


start = time.time()

df = pd.read_csv('../datasets/arth150.csv', sep = ';')
arr = np.array(df)
scaler = preprocessing.StandardScaler().fit(arr)
arr = scaler.transform(arr)


colnames = list(df.columns.values.tolist())
arcs = pd.read_csv('../datasets/arth150_arcs.csv', sep = ';')

arcs = np.array(arcs,dtype="str")
cols = {}
for i in range(len(colnames)):
    cols[colnames[i]] = i

arcs_num = []
for arc in arcs:
    arcs_num.append([cols[arc[0]], cols[arc[1]]])

dag_true = dag_to_mat_true(arcs_num, colnames)
np.savetxt("../results/true_dag_arth150.csv",dag_true,delimiter =";")

# statistics, result = experiment(arr, 46, dag_true, sizes, b_start_coef_list=[0.5,0.1,0.05,0.005, 0.0005], g_start_coef_list=[0.5,0.1,0.05,0.005, 0.0005], 
# decay_bic_list=[0], decay_gic_list =[0],
#  penalty_bic_decay_pattern = division_decay_bic, 
#  penalty_gic_decay_pattern =division_decay,filename= 'different_decay.csv',N=20)

statistics, result, layers, dag_est1, sample = experiment(arr, 107, dag_true, sizes, b_start_coef_list=[0.5], g_start_coef_list=[0.5], 
    decay_bic_list=[0], decay_gic_list =[0],
    penalty_bic_decay_pattern = division_decay, 
    penalty_gic_decay_pattern =division_decay,filename= 'same_decay.csv',N=100)


