from typing import List, Optional, Dict

import numpy as np


class FlowGraph:
    def __init__(
            self,
            nodes: List[int],
            source: int,
            sink: int,
            capacities: np.ndarray,
            meta: Optional[Dict] = None):

        self.nodes = nodes
        self.source = source
        self.sink = sink
        self.edges = (capacities > 0).astype(int)
        self.capacities = capacities
        self.meta = meta

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


class Grid:
    def __init__(
        self,
        capacities: np.ndarray,
        source_idx: int = 0,
        sink_idx: Optional[int] = None):

        if not capacities.ndim == 2:
            raise ValueError("Capacities must be a square matrix.")
        if not capacities.shape[0] == capacities.shape[1]:
            raise ValueError("Capacities must be a square matrix.")
        if not (capacities >= 0).all():
            raise ValueError("Capacities must be positive.")

        self.capacities = capacities
        self.source_idx = source_idx
        self.sink_idx = sink_idx if sink_idx else self.num_nodes - 1

    @property
    def num_nodes(self) -> int:
        return self.capacities.shape[0]


class MinimalCut:
    def __init__(self):
        pass

