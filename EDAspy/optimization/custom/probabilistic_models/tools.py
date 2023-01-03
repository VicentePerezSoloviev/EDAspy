#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def arcs2adj_mat(arcs: list, n_variables: int) -> np.array:
    """
    This function transforms the list of arcs in the BN structure to an adjacency matrix.
    :param arcs: list of arcs in the BN structure.
    :param n_variables: number of variables.
    :type arcs: list
    :type n_variables: int
    :return: adjacency matrix
    :rtype: np.array
    """

    matrix = np.zeros((n_variables, n_variables))
    for arc in arcs:
        matrix[arc[0], arc[1]] = 1

    return matrix


def _noise(n_variables: int, size: float) -> np.array:
    h_noise = np.zeros(n_variables)
    h_noise[::2] = size*2
    return h_noise - size


def _set_positions(variables: list) -> dict:
    n_variables = len(variables)
    n_cols = int(np.sqrt(n_variables))
    n_rows = int(np.ceil(n_variables / n_cols))

    pos_list = []
    for row in range(n_rows):
        for col in range(n_cols):
            pos_list.append([col, -row])

    '''if noise:
        noise_list = _noise(len_variables, size)
        for i in range(len_variables):
            pos_list[i][0] += noise_list[i]
            pos_list[i][1] += noise_list[i]'''

    pos = {}
    for i in range(n_variables):
        pos.update({variables[i]: pos_list[i]})

    return pos


def plot_bn(arcs: list, n_variables: int, pos: dict = None, curved_arcs: bool = True,
            curvature: float = -0.3, node_size: int = 500, node_color: str = 'red',
            edge_color: str = 'black', arrow_size: int = 15, node_transparency: float = 0.9,
            edge_transparency: float = 0.9, node_line_widths: float = 2, title: str = None,
            output_file: str = None):
    """
    This function Plots a BN structure as a directed acyclic graph.
    :param arcs: Arcs in the BN structure.
    :param n_variables: Number of variables in the BN structure.
    :param pos: Positions in the plot for each node.
    :param curved_arcs: True if curved arcs are desired.
    :param curvature: Radians of curvature for edges. By default, -0.3.
    :param node_size: Size of the nodes in the graph. By default, 500.
    :param node_color: Color set to nodes. By default, 'red'.
    :param edge_color: Color set to edges. By default, 'black'.
    :param arrow_size: Size of arrows in edges. By default, 15.
    :param node_transparency: Alpha value [0, 1] that defines the transparency of the node. By default, 0.9.
    :param edge_transparency: Alpha value [0, 1] that defines the transparency of the edge. By default, 0.9.
    :param node_line_widths: Width of the nodes contour lines. By default, 2.0.
    :param title: Title for Figure. By default, None.
    :param output_file: Path to save the figure locally.
    :type arcs: List of tuples.
    :type n_variables: int.
    :type pos: dict with name of variables as keys and tuples with coordinates as values.
    :type curved_arcs: bool.
    :type curvature: float.
    :type node_size: int.
    :type node_color: str.
    :type edge_color: str.
    :type arrow_size: int.
    :type node_transparency: float.
    :type edge_transparency: float.
    :type node_line_widths: float.
    :type title: str.
    :type output_file: str.
    :return: Figure.
    """

    nodes = [str(i) for i in range(n_variables)]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(arcs)

    if not pos:
        pos = _set_positions(nodes)

    nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color=node_color, alpha=node_transparency,
                           linewidths=node_line_widths)

    if curved_arcs:
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color=edge_color,
                               connectionstyle="arc3,rad=" + str(curvature), arrowsize=arrow_size,
                               alpha=edge_transparency)
    else:
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color=edge_color, arrowsize=arrow_size)

    nx.draw_networkx_labels(g, pos)

    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)

    plt.show()
