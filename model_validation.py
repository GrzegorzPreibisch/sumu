import sumu
import pandas as pd
import numpy as np
import random
import time
import networkx as nx


def dag_generator(dim, chidren_num, parents_num):
    possible_children = set(range(0, dim))
    possible_parents = set()
    roots = (list(set(random.choices(list(possible_children), k=chidren_num))))
    for root in roots:
        possible_children.remove(root)
        possible_parents.add(root)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(0, dim))

    for x in range(0, dim):
        if len(possible_children) != 0:
            new_children = list(set(random.choices(list(possible_children), k=chidren_num)))
        else:
            break
        their_parents = list(set(random.choices(list(possible_parents), k=parents_num)))

        for child in new_children:
            possible_parents.add(child)
            possible_children.remove(child)
            for parent in their_parents:
                dag.add_edge(parent, child)
    delete_edges = list(set(random.choices(list(dag.edges), k=dim)))
    for edge in delete_edges:
        dag.remove_edge(edge[0], edge[1])

    return dag


def generate_dataset(dim, number):
    dictionary = {}
    for x in range(0, dim):
        dictionary[x] = np.random.normal(1, 1, number)
    df = pd.DataFrame(dictionary)
    return df


def create_dependencies(df, dag):
    topology = list(nx.topological_sort(dag))
    relations = dict()
    for x in list(dag.nodes):
        relations[x] = []
    for e in list(dag.edges):
        relations[e[0]].append(e[1])
    for node in topology:
        for relation in relations[node]:
            df[relation] += df[node]
    return df


def create_dataset(dim, number, chidren_num, parents_num):
    dag = dag_generator(dim, chidren_num, parents_num)
    df = generate_dataset(dim, number)
    new_df = create_dependencies(df, dag)
    return np.array(new_df), dag


test_arr, test_dag = create_dataset(25, 5000000, 4, 3)

start = time.time()
data = sumu.Data(test_arr, discrete=False)
params = {

    "array": test_arr,
    "data": data,
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
dag = g.sample()
end = time.time()


def dag_validation(dag, result):

    relations = dict()
    for x in list(dag.nodes):
        relations[x] = []
    for e in list(dag.edges):
        relations[e[0]].append(e[1])
    relations_result = dict()
    for x in list(dag.nodes):
        relations_result[x] = []
    for x in relations:
        if x in result:
            if len(result[x]) != 0:
                for y in result[x]:
                    relations_result[x].append(y[0])

    count_all = 0
    count_hits = 0
    for x in relations:
        for y in relations[x]:
            count_all += 1
            if y in relations_result[x]:
                count_hits += 1
    return count_all, count_hits


count_all, count_hits = dag_validation(test_dag, dag)
print(end-start)
print(count_all, count_hits)
