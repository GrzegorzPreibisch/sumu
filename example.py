import numpy as np
import sumu
import time

def generate_dag(size, lam = 2, L = 10 ):
    order = np.random.choice(size,L,replace=False)
    order[L-1] = size-1 
    possible_parents = []
    dag = {}
    i = 0
    j = 0
 
    for i in range(1,L):
      while j<= order[i]:
        if j>order[0]:          
          number_of_parents =  min(order[i-1],np.random.poisson(lam-1)+1)
          parents = np.random.choice(order[i-1],number_of_parents,replace=False)
          dag[j] = parents
        j+=1
        
    return dag       
 

def  generate_data(n ,p, dag):
    data = np.random.normal(1,1,(n,p))
    for el in dag:
        for k in dag[el]:
             data[:,el]+=data[:,k]
    return data

def dag_to_mat(dag,p,est=True):
    mat = np.zeros((p,p))
    for i in range(p):
       if i in dag:
         for x in dag[i]:
            if est:
                mat[i,x[0]] = 1
            else:
                mat[i,x] =1
    return mat 

def compute_stats(dag_est,dag_true,p):
    mat_t = dag_to_mat(dag_true,p,False)
    mat_e = dag_to_mat(dag_est,p,True)
    TP =  np.sum(mat_t*mat_e)
    WD =  np.sum(mat_t*mat_e.transpose())
    FP =  -np.sum((mat_t-1)*mat_e)
    TN =  np.sum((mat_t-1)*(mat_e-1))
    FN =   -np.sum((mat_t)*(mat_e-1))
    return TP,FP,TN,FN,WD, np.sum(mat_t),np.sum(mat_e)
n = 50
p = 20

dag = generate_dag(p,L=5)
data = generate_data(n,p,dag) 
data_sumu = sumu.Data(data,discrete =False)          
 
params = {

    "array":data,
    "data": data_sumu,
    "scoref": "bge",  # Or "bdeu" for discrete data.
    # "ess": 10,        # If using BDeu.
    "max_id": -1,
    "K": 5,
    "d": 5,
    "cp_algo": "greedy-lite",
    "mc3_chains": 15,
    "burn_in": 10,
    "iterations": 5000,
    "thinning": 10}

g = sumu.Gadget(**params)
h = g.sample()
for c in range(1,50):
 dag_est1, intercept = h.generate_final_dag(pen_bic= np.log(n),pen_gic=c*np.log(p))
 print(c,compute_stats(dag_est1,dag,p))
end = time.time()




