from typing import Tuple


class AStarNode:
    """
    A node for A*.
    """

    def __init__(
        self, grid: Tuple[int, ...], parent: Tuple[int, ...], g: int, h: float
    ):
        self.grid = grid
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
        self.hash = hash(grid)

    def __hash__(self) -> int:
        return self.hash

    def __lt__(self, other_node) -> bool:
        return (
            self.g > other_node.g if self.f == other_node.f else self.f < other_node.f
        )

    def __eq__(self, other: "AStarNode") -> bool:
        return self.grid == other.grid


class NoTieBreakAStarNode:
    """
    A node for A* without tie-breaking.
    """

    def __init__(
        self, grid: Tuple[int, ...], parent: Tuple[int, ...], g: int, h: float
    ):
        self.grid = grid
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
        self.hash = hash(grid)

    def __hash__(self) -> int:
        return self.hash

    def __lt__(self, other_node):
        return self.f < other_node.f

    def __eq__(self, other: "AStarNode") -> bool:
        return self.grid == other.grid


class OneDimBucketList:
    """
    Implementation of a 1D bucket list.
    """

    def __init__(self, size=0):
        self.bins = [[] for _ in range(size)]  # Indexation by f value.
        self.min_indice = -1  # The minimal f value in the list.
        self.length = 0

    def __len___(self) -> int:
        return self.length

    def empty(self) -> bool:
        return self.length == 0

    def add(self, node: NoTieBreakAStarNode) -> None:
        """
        Adds the node in the list.
        """
        if node.f >= len(self.bins):
            for _ in range(node.f - len(self.bins) + 1):
                self.bins.append([])
        self.bins[node.f].append(node)
        self.length += 1
        if self.min_indice == -1 or node.f < self.bins[self.min_indice][0].f:
            self.min_indice = node.f

    def update(
        self,
        node: NoTieBreakAStarNode,
        new_f: int,
        new_g: int,
        new_parent: NoTieBreakAStarNode,
    ) -> None:
        """
        Updates the attributes of a node and reposition
        it in the list.
        """
        i = self.bins[node.f].index(node)
        node = self.bins[node.f][i]  # We do this to keep the reference to the node
        self.bins[node.f].pop(i)
        node.f = new_f
        node.g = new_g
        node.parent = new_parent
        self.bins[node.f].append(node)

    def pop(self) -> NoTieBreakAStarNode:
        """
        Pop a node with least f value.
        """
        self.length -= 1
        min_list = self.bins[self.min_indice]
        min_node = min_list.pop()
        if self.length == 0:
            self.min_indice = -1
            return min_node
        for i in range(self.min_indice, len(self.bins)):
            if self.bins[i]:
                break
        self.min_indice = i
        return min_node


class TwoDimBucketList:
    """
    Implementation of a 1D bucket list.
    """

    def __init__(self, size1=0, size2=0):
        self.bins = [
            [[] for _ in range(size2)] for _ in range(size1)
        ]  # Indexation by f value then g value.
        self.min_f_index = -1  # The minimum f value in the list.
        self.max_g_indices = []  # The maximum g values for each f value.
        self.length = 0

    def __len___(self) -> int:
        return self.length

    def empty(self) -> bool:
        return self.length == 0

    def update(
        self, node: AStarNode, new_f: int, new_g: int, new_parent: AStarNode
    ) -> None:
        """
        Updates the attributes of a node and reposition
        it in the list.
        """
        f_index = node.f
        i = self.bins[f_index][node.g].index(node)
        node = self.bins[f_index][node.g].pop(
            i
        )  # We do this to keep the reference to the node.
        self.length -= 1
        self.update_indices(f_index, node.g)
        node.f = new_f
        node.g = new_g
        node.parent = new_parent
        self.add(node)

    def add(self, node: "AStarNode") -> None:
        """
        Adds the node in the list.
        """
        if node.f >= len(self.bins):
            for _ in range(node.f - len(self.bins) + 1):
                self.bins.append([])
                self.max_g_indices.append(-1)
        if node.g >= len(self.bins[node.f]):
            for _ in range(node.g - len(self.bins[node.f]) + 1):
                self.bins[node.f].append([])
        if (
            self.min_f_index == -1
            or node
            < self.bins[self.min_f_index][self.max_g_indices[self.min_f_index]][0]
        ):
            self.min_f_index = node.f
        self.max_g_indices[node.f] = max(node.g, self.max_g_indices[node.f])
        self.bins[node.f][node.g].append(node)
        self.length += 1

    def update_indices(self, f_index: int, g_index: int) -> None:
        """
        Recomputes the minimum f value and maximal g values after
        a deletion.

        Parameters:
        -----------
        f_index: int
            The f value of the node who was removed.
        g_index: int
            The g value of the node who was removed.
        """
        if self.length == 0:
            self.max_g_indices[f_index] = -1
            self.min_f_index = -1
            return
        if self.bins[f_index][g_index] or g_index < self.max_g_indices[f_index]:
            # If the bin isn't empty or isn't the bin with maximal g value.
            return
        for k in range(g_index - 1, -1, -1):
            if self.bins[f_index][k]:
                # We search the non-empy bins with maximal g value.
                self.max_g_indices[f_index] = k
                return
        self.max_g_indices[f_index] = -1
        if f_index == self.min_f_index:
            for j in range(f_index + 1, len(self.bins)):
                if self.max_g_indices[j] > 0:
                    self.min_f_index = j
                    return

    def pop(self) -> AStarNode:
        """
        Pop a node with maximal g value among the nodes with minimal f value.
        """
        self.length -= 1
        i, j = self.min_f_index, self.max_g_indices[self.min_f_index]
        min_node = self.bins[i][j].pop()
        self.update_indices(i, j)
        return min_node
