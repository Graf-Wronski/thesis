import dataclasses
from typing import List, Optional, Dict

import numpy as np


class FlowNetwork:
    def __init__(
            self,
            vertices: List[int],
            source: int,
            sink: int,
            capacities: np.ndarray,
            meta: Optional[Dict] = None):

        self.vertices = vertices
        self.source = source
        self.sink = sink
        self.edges = (capacities > 0).astype(int)
        self.capacities = capacities
        self.meta = meta

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)


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


@dataclasses.dataclass
class Cut:
    # A cut partitions the graph in two parts.

    # We are only interested in partitions that seperate source and sink.
    v_source: list[int]
    v_sink: list[int]

    capacities: np.ndarray

    @property
    def value(self) -> float:
        return self.capacities.sum()

    @property
    def edges(self) -> np.ndarray:
        return self.capacities > 0.