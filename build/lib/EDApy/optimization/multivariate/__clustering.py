#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def clustering(n_clusters, dataset, evidences, cluster_vars):
    cluster_data = pd.DataFrame(columns=cluster_vars)
    data = []

    for i in cluster_vars:
        data.append(list(dataset[i]))

    values = list(dataset.index)

    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(data).T)
    labels = list(k_means.labels_)

    simulation = []
    for i in cluster_vars:
        for j in evidences:
            if i == j[0]:
                simulation.append(j[1])

    assignment_cluster = k_means.predict([simulation])[0]

    index = 0
    for i in labels:
        if i == assignment_cluster:
            cluster_data = cluster_data.append(dataset.loc[values[index]])
        index = index + 1

    return cluster_data
