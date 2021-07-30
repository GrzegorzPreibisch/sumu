import bnlearn
import numpy as np
import sumu
import time
import bnlearn as bn

start = time.time()
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

def dag_dict_to_list(dag: dict, model=True):
    g = dict()
    if model:
        for key in dag:
            temp_list = list()
            for tuple in dag[key]:
                temp_list.append(tuple[0])
            g[key] = list(temp_list)
    else:
        for key in dag:
            g[key] = list(dag[key])
    dag_list = []
    for node in g:
        for edge in g[node]:
            dag_list.append((node, edge))

    return dag_list


def final_dag_to_adj_matrix(g: dict, p:int , intercept:dict, key_to_id=None)->np.array:

    if p !=  len(g.keys()):
        print('We do not save all edges, p!= len(keys)')

    M = np.zeros((p,p))

    for parent in g.keys():
        for child, est in g[parent]:
            if key_to_id != None:
                M[key_to_id[parent],key_to_id[child]]=est
            else:
                M[parent, child] = est
    for edge in intercept:
        if key_to_id != None:
            M[key_to_id[edge], key_to_id[edge]] = intercept[edge]
        else:
            M[edge, edge] = intercept[edge]
    return M


n = 20
p = 10

dag = generate_dag(p,L=7)
data = generate_data(n,p,dag)
data_sumu = sumu.Data(data)


g = sumu.Gadget(data=data_sumu)
g.sample()
for c in range(1,3):
 dag_est1, intercept = g.generate_final_dag(pen_bic= np.log(n),pen_gic=c*np.log(p))
 print(c,compute_stats(dag_est1,dag,p))
 print(intercept)

print(intercept)

end = time.time()
print(dag)


ground_truth = bnlearn.make_DAG(dag_dict_to_list(dag, model=False))
print(dag_est1)

final_dag_matrix =final_dag_to_adj_matrix(dag_est1,p,intercept)

np.savetxt('dag.txt',final_dag_matrix)

model =  bnlearn.make_DAG(dag_dict_to_list(dag_est1,model=True))


print(end - start)
bn.compare_networks(ground_truth, model)


