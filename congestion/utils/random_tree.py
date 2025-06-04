import numpy as np


class RandomTree:
    def __init__(self, n_grid_loads: int = 10, n_feeders: int = 2):

        """ A RandomTree that serves as sample radial grid topology.
        Args:
            n_grid_loads: The number of grid nodes, i.e. the number of tree
                leaves. (Which is smaller than the number of tree nodes!)
            n_feeders: The number of feeders, i.e. the number of the root
                nodes' children.
        """

        if not isinstance(n_grid_loads, int) or n_grid_loads <= 0:
            raise ValueError("n_grid_nodes must be an integer greater than 0.")

        if not isinstance(n_feeders, int) or n_feeders <= 0:
            raise ValueError("n_feeders must be an integer greater than 0.")

        if n_grid_loads < n_feeders:
            raise ValueError("n_grid_nodes cannot be less than n_feeders.")

        self.n_grid_loads = n_grid_loads
        self.n_feeders = n_feeders

        # All trees are rooted in 0.
        self.root = 0
        self.nodes = [0]
        self.edges = []
        self.leaves = []
        self.leave_probabilities = []

        # Create initial feeders.
        for idx in range(0, self.n_feeders):
            feeder = idx + 1  # 0 = 'root'
            self.nodes.append(feeder)
            self.edges.append([self.root, feeder])
            self.leaves.append(feeder)
            # Initially all leaves have the same probability.
            self.leave_probabilities.append(1 / self.n_feeders)

        while self.n_leaves < self.n_grid_loads:
            # Pick a leave.
            leaf_idx = np.random.choice(
                a=np.arange(self.n_leaves),
                p=self.leave_probabilities)
            leaf = self.leaves[leaf_idx]
            p_leaf = self.leave_probabilities[leaf_idx]

            # Sample children: not more than 3 and not more than missing nodes.
            self.max_children = min(3, self.n_grid_loads - self.n_leaves + 1)
            n_children = np.random.choice(np.arange(2, self.max_children + 1))

            # Add children to tree.
            for child in range(self.n_nodes, self.n_nodes + n_children):
                self.nodes.append(child)
                self.edges.append([leaf, child])
                # Children are leaves. Probability is shared among children.
                self.leaves.append(child)
                self.leave_probabilities.append((p_leaf / n_children))

            # Leaf is now an inner node.
            del self.leaves[leaf_idx]
            del self.leave_probabilities[leaf_idx]

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)