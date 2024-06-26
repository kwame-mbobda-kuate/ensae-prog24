from grid import Grid
import pickle
import numpy as np
import utils
import gzip
import utils
import time
from typing import List, Dict, Tuple, Callable, Any, Union
import collections


class APDB:
    """
    An abstract class used to represent additive pattern database (APDB) for the swap puzzle as
    described in https://arxiv.org/pdf/1107.0050.pdf.
    """

    def __init__(
        self, m: int, n: int, group: Tuple[int, ...], apdb: Any = None
    ) -> None:
        """
        Parameters:
        -----------
        m, n: int
            The number of lines and colums of the grids which will be stored.
        group: Tuple[int, ...]
            The tuple of integers defining the pattern group.
        apdb: Any
            The database.
        """
        self.m = m
        self.n = n
        self.group = group
        self.apdb = apdb

    def load(cls, filename: str) -> "APDB":
        """
        Loads an APDB stored.
        """
        raise NotImplementedError

    def save(self, filename: str) -> None:
        """
        Saves an APDB.
        """
        raise NotImplementedError

    def compute(self) -> None:
        """
        Computes the APDB.
        """
        raise NotImplementedError

    def heuristic(self, grid: Tuple[int, ...]) -> float:
        """
        Computes the heuristic value of a grid.
        """
        raise NotImplementedError


def representation(group: Tuple[int, ...], grid: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes the group representation of a grid.

    Parameters:
    -----------
    grid: Tuple[int, ...]
        The grid.

    Output:
    -------
    tuple: Tuple[int, ...]
        The group representation of the grid. It contains
        the indices where the numbers of the pattern
        groups are.
    """
    return tuple(grid.index(k, 2) - 2 for k in group)


class ArrayAPDB(APDB):
    """
    An implementation of APDB using np.array.
    """

    def compute(self) -> None:
        self.apdb = np.zeros([self.m * self.n] * len(self.group), dtype=np.int8) - 1
        # Using np.int8 to save space
        # -1 means a grid hasn't been visited
        src = list(Grid(self.m, self.n).to_tuple())
        for k in range(2, self.m * self.n + 2):
            if src[k] not in self.group:
                src[k] = -1
        src = tuple(src)
        all_swaps = Grid.all_swaps(self.m, self.n)
        self.apdb[representation(self.group, src)] = 0
        queue = collections.deque([(src, 0)])
        while queue:
            node, d = queue.pop()
            L = list(node)
            for swap in all_swaps:
                i, j = (
                    swap[0][0] * self.n + swap[0][1] + 2,
                    swap[1][0] * self.n + swap[1][1] + 2,
                )
                if L[i] == L[j] == -1:
                    continue
                # We don't consider a swap between two -1 tiles
                utils.make_swap(L, swap)
                neighbor = tuple(L)
                utils.make_swap(L, swap)
                repr = representation(self.group, neighbor)
                if self.apdb[repr] < 0:
                    queue.appendleft((neighbor, d + 1))
                    self.apdb[repr] = d + 1

    @classmethod
    def load(cls, filename: str) -> "ArrayAPDB":
        with np.load(
            filename + (".npz" if not filename.endswith(".npz") else ""), "rb"
        ) as f:
            return ArrayAPDB(f["m"], f["n"], f["group"], f["apdb"])

    @staticmethod
    def default_filename(m: int, n: int, group: Tuple[int, ...]) -> str:
        return f"apdb\\{m} x {n}, {' '.join([str(k) for k in group])}.npz"

    @classmethod
    def default_load(cls, m: int, n: int, group: Tuple[int, ...]) -> "ArrayAPDB":
        return ArrayAPDB.load(ArrayAPDB.default_filename(m, n, group))

    def save(self, filename="") -> None:
        if self.apdb == []:
            self.compute()
        filename = filename or ArrayAPDB.default_filename(self.m, self.n, self.group)
        np.savez_compressed(
            filename, apdb=self.apdb, m=self.m, n=self.n, group=self.group
        )

    def heuristic(self, grid) -> int:
        return self.apdb[representation(self.group, grid)]


class DictAPDB(APDB):
    """
    An implementation of APDB using dictionnaries.
    """

    def mapping(self, grid: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(grid.index(k, 2) - 2 for k in self.group)

    @staticmethod
    def default_filename(m: int, n: int, group: Tuple[int, ...]) -> str:
        return f"apdb\\{m} x {n}, {' '.join([str(k) for k in group])}"

    def compute(self):
        src = list(Grid(self.m, self.n).to_tuple())
        for k in range(2, self.m * self.n + 2):
            if src[k] not in self.group:
                src[k] = -1
        src = tuple(src)
        all_swaps = Grid.all_swaps(self.m, self.n)
        self.apdb = {src: 0}
        queue = collections.deque([src])
        while queue:
            node = queue.pop()
            d = self.apdb[node]
            L = list(node)
            for swap in all_swaps:
                i, j = (
                    swap[0][0] * self.n + swap[0][1] + 2,
                    swap[1][0] * self.n + swap[1][1] + 2,
                )
                if L[i] == L[j] == -1:
                    continue
                utils.make_swap(L, swap)
                cpy = tuple(L)
                utils.make_swap(L, swap)
                if cpy not in self.apdb:
                    queue.appendleft(cpy)
                    self.apdb[cpy] = d + 1
        # The group representation is far more compact than the usual one
        self.apdb = {
            representation(self.group, grid): d for grid, d in self.apdb.items()
        }

    def save(self, filename="") -> None:
        if not self.apdb:
            self.compute()
        filename = filename or DictAPDB.default_filename(self.m, self.n, self.group)
        with open(filename, "wb") as f:
            f.write(gzip.compress(pickle.dumps(self)))

    @classmethod
    def load(cls, filename) -> "DictAPDB":
        with open(filename, "rb") as f:
            return pickle.loads(gzip.decompress(f.read()))

    @classmethod
    def default_load(cls, m: int, n: int, group: Tuple[int, ...]) -> "DictAPDB":
        return DictAPDB.load(DictAPDB.default_filename(m, n, group))

    def heuristic(self, grid: Tuple[int, ...]) -> int:
        return self.apdb[representation(self.group, grid)]


class VertSymmetryAPDB:
    """
    Represents the result of a vertical symmetry on a APDB.
    """

    def __init__(self, apdb: "APDB"):
        self.apdb = apdb

    def heuristic(self, grid: Tuple[int, ...]) -> int:
        m, n = grid[0], grid[1]
        symmetry = list(grid)
        for i in range(m):
            for j in range(n):
                y, x = divmod(grid[n * i + j + 2] - 1, n)
                symmetry[n * (m - 1 - i) + j + 2] = n * (m - 1 - x) + y + 1
        return self.apdb.heuristic(symmetry)


class APDBList:
    """
    A class used to aggregate different APDBs.
    """

    def __init__(
        self, f: Callable[[List[float]], float], apdbs: List[Union["APDB", "APDBList"]]
    ):
        self.f = f  # f will usually be the max or the sum
        self.apdbs = apdbs

    def heuristic(self, grid: Tuple[int, ...]) -> float:
        return self.f([apdb.heuristic(grid) for apdb in self.apdbs])
