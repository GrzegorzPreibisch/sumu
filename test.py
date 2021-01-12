import sumu
import pandas as pd
import numpy as np

from glmnet import ElasticNet

import time
# import glmnet_python
# from glmnet import glmnet

start = time.time()
df = pd.read_csv('test_data.csv')

arr = np.array(df)

data = sumu.Data(arr, discrete=False)

params = {

          "array": arr,
          "data": data,
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
dag = g.sample()
end = time.time()
print(dag)

print(end - start)
