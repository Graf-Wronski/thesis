from typing import Set
import numpy as np


def is_connected(adj_matrix: np.ndarray, start_vertex: int = 0) -> bool:
    """ Perform a breadth-first search to check if a graph is connected.

    Args:
        adj_matrix: The adjacency matrix of the graph.
        start_vertex: The starting vertex of the breadth-first search.

    Returns:
        bool: True if the graph is connected, False otherwise.
    """

    if not adj_matrix.shape[0] == adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    reachables_vertices = get_reachable_vertices(adj_matrix, start_vertex)

    # If all vertices are reachable, then the graph is connected.
    if len(reachables_vertices) == adj_matrix.shape[0]:
        return True
    else:
        return False


def get_reachable_vertices(adj_matrix: np.ndarray, start: int) -> list[int]:
    """ All vertices reachable from start. """

    vertices = [x for x in range(adj_matrix.shape[0])]
    visited = {x: False for x in vertices}

    # To_visit is a set as we do not visit a single vertex twice.
    to_visit = {start}

    def visit(vertex: int, future_visits: Set[int]):
        """ Visit a vertex and regard connected vertices if they are unvisited. """
        visited[vertex] = True
        connected_vertices = np.nonzero(adj_matrix[vertex, :])

        for x in list(*connected_vertices):
            if not (visited[x]):
                future_visits.add(x)

    while len(to_visit) > 0:
        elem = to_visit.pop()
        visit(elem, to_visit)

    return [v for v in vertices if visited[v]]
    


