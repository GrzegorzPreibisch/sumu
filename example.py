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

n = 50
p = 30

dag = generate_dag(p)
data = generate_data(n,p,dag) 
data_sumu = sumu.Data(data,discrete =False)          

params = {

    "array":data,
    "data": data_sumu,
    "scoref": "bge",  # Or "bdeu" for discrete data.
    # "ess": 10,        # If using BDeu.
    "max_id": -1,
    "K": 7,
    "d": 5,
    "cp_algo": "greedy-lite",
    "mc3_chains": 15,
    "burn_in": 10,
    "iterations": 50000,
    "thinning": 10}

g = sumu.Gadget(**params)
h = g.sample()
dag_est1 = h.generate_final_dag(pen_bic= np.log(n),pen_gic=2*np.log(p))
dag_est2 = h.generate_final_dag(np.log(n),4*np.log(p))
end = time.time()


