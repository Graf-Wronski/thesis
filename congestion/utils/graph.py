from typing import Set

import numpy as np


def is_connected(adj_matrix: np.ndarray, start_node: int = 0) -> bool:
    """ Perform a breadth-first search to check if a graph is connected.

    Args:
        adj_matrix: The adjacency matrix of the graph.
        start_node: The starting node of the breadth-first search.

    Returns:
        bool: True if the graph is connected, False otherwise.
    """

    if not adj_matrix.shape[0] == adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    nodes = [x for x in range(adj_matrix.shape[0])]
    visited = {x: False for x in nodes}
    # To_visit is a set as we do not visit a single node twice.
    to_visit = {start_node}

    def visit(node: int, future_visits: Set[int]):
        """ Visit a node and regard connected nodes if they are unvisited. """
        visited[node] = True
        connected_nodes = np.nonzero(adj_matrix[node, :])

        for x in list(*connected_nodes):
            if not(visited[x]):
                future_visits.add(x)

    while len(to_visit) > 0:
        elem = to_visit.pop()
        visit(elem, to_visit)

    # If all nodes were visited, then the graph is connected.
    if len([x for x in nodes if visited[x]]) == len(nodes):
        return True
    else:
        return False



