import sumu
import pandas as pd
import numpy as np
import time
# import glmnet_python
# from glmnet import glmnet

from glmnet import ElasticNet

start = time.time()
df = pd.read_csv('test_data.csv')

arr = np.array(df)

data = sumu.Data(arr, discrete=False)

params = {"data": data,
          "scoref": "bge",  # Or "bdeu" for discrete data.
          # "ess": 10,        # If using BDeu.
          "max_id": -1,
          "K": 7,
          "d": 10,
          "cp_algo": "greedy-lite",
          "mc3_chains": 10,
          "burn_in": 100,
          "iterations": 5000,
          "thinning": 10}

g = sumu.Gadget(**params)
dag, score = g.sample()
end = time.time()


previous_parent = list(dag[0])
final_dag = dict()
for i in range(1, len(dag)):
    for j in dag[i]:
        # code fro regression
        y = arr[:, j]
        x = np.zeros((arr.shape[0], len(previous_parent)))
        for col in range(len(previous_parent)):
            x[:, col] = arr[:, previous_parent[col]]
        m = ElasticNet()
        m = m.fit(x, y)
        #print(m.coef_path_)
        final_dag[j] = list()
        for v in range(len(m.coef_)):
            if m.coef_[v] !=0:
                final_dag[j].append((previous_parent[v],m.coef_[v])) ## If intersept is 0 and changes order
    # print(previous_parent, j)
    previous_parent = list(set(previous_parent).union(dag[i]))

print(final_dag)

# The following only for continuous data
# dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]
# causal_effects = sumu.beeps(dags, data)
# print(causal_effects)

print(end - start)
