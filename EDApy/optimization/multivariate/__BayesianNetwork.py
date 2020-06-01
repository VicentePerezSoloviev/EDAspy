#!/usr/bin/env python
# coding: utf-8

from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
import numpy as np
from pyvis.network import Network
import random

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
bnlearnPac = rpackages.importr("bnlearn")


def learn_structure(dataset, algorithm, black_list):
    pandas2ri.activate()

    # dataframes R
    r_dt = pandas2ri.py2ri(dataset)
    r_blacklist = pandas2ri.py2ri(black_list)

    if algorithm == 'rsmax2':
        hpc, rsmax2, tabu = bnlearnPac.hpc, bnlearnPac.rsmax2, bnlearnPac.tabu
        structure = rsmax2(r_dt, restrict='hpc', maximize='tabu', blacklist=r_blacklist)

    elif algorithm == 'mmhc':
        mmhc = bnlearnPac.mmhc
        structure = mmhc(r_dt, blacklist=r_blacklist)

    elif algorithm == 'hc':
        hc = bnlearnPac.hc
        structure = hc(r_dt, blacklist=r_blacklist)

    else:
        raise Exception('Learning algorithm not allowed')

    return structure


def calculate_fit(data, algorithm, black_list):
    bn_fit = bnlearnPac.bn_fit
    structure = learn_structure(data, algorithm, black_list)
    fithistorico = bn_fit(structure, data)
    return fithistorico


def print_structure(structure, var2optimize, evidences):
    adjacency_matrix = bnlearnPac.amat(structure)  # adjacency matrix
    vector = np.array(adjacency_matrix)
    nodes = bnlearnPac.nodes(structure)  # nodes names
    positions = []
    mod = 600
    p = 0

    lista = var2optimize + evidences + list(set(nodes) - set(var2optimize + evidences))
    for i in lista:
        positions.append([i, int(p / mod) * 200 + (random.randint(-20, 20)), p % mod + (random.randint(-20, 20))])
        p = p + 150

    color_map = []
    for i in list(nodes):
        if i in var2optimize:
            color_map.append('red')
        if i in evidences:
            color_map.append('blue')
        if i == 'COSTE':
            color_map.append('green')

    Gr = Network("800px", "800px", directed=True)
    Gr.add_nodes(list(range(len(nodes))), title=nodes,
                 x=[row[1] for row in positions], y=[row[2] for row in positions],
                 color=color_map, label=nodes)

    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j] == 1:
                Gr.add_edge(list(range(len(nodes)))[i], list(range(len(nodes)))[j])

    for n in Gr.nodes:
        n.update({'physics': False})

    # Gr.show_buttons(filter=['physics'])
    Gr.show_buttons()
    Gr.show("mygraph.html")

