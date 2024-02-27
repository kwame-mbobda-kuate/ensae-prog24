from grid import Grid
import copy
import itertools
import pickle
import os
import pickle
import gzip
import numpy as np
import utils
import math
from typing import List, Dict, Tuple, Callable
import collections
import time


def are_disjoint(l1, l2):
    for i in l1:
        if i in l2:
            return False
    return True


def half_manhattan_distance(length: int, n: int, grid: Tuple[int, ...]) -> float:
    """
    Compute the half sum of the Manhattan distances between the cell occupied by each number
    in a given grid and the cell it should occupy in the sorted grid.
    The representation used here is different from the usual one.
    """
    return (
        sum(
            abs(((grid[i + length] - 1) % n - (grid[i] % n)))
            + abs((grid[i + length] - 1) // n - grid[i] // n)
            for i in range(length)
        )
        / 2
    )


def get_neighbors(m: int, n: int, grid) -> Tuple[int, ...]:
    """
    Determine the grids reachable in a swap from a given grid.
    The representation used here is different from the usual one.
    """
    neighbors = []
    N = len(grid) // 2
    coords = grid[:N]
    for i in range(N):
        for j in range(i):
            if (abs(grid[i] - grid[j]) == 1 and grid[i] // n == grid[j] // n) or abs(
                grid[i] - grid[j]
            ) == n:
                L = list(grid)
                L[i], L[j] = L[j], L[i]
                neighbors.append(tuple(L))
    for i in range(N):
        for j in (-1, 1):
            if (
                grid[i] // n == (grid[i] + j) // n
                and 0 <= grid[i] + j < m * n
                and grid[i] + j not in coords
            ):
                L = list(grid)
                L[i] += j
                neighbors.append(tuple(L))
            if 0 <= grid[i] + j * n < m * n and grid[i] + j * n not in coords:
                L = list(grid)
                L[i] += j * n
                neighbors.append(tuple(L))
    return neighbors


class GADB:

    def __init__(self, m: int, n: int, k: int, gadb: Dict[Tuple[int, ...], float] = {}):
        self.m = m
        self.n = n
        self.k = k
        self.gadb = gadb

    def compute(self):
        """
        Computes a General Additive Database (or also called dynamically-partitionned database)
        as described in https://arxiv.org/pdf/1107.0050.pdf. This functions finds all
        the k-uplets of tiles whose distance is superior than their half Mahnattan distance.
        """
        self.gadb = dict()
        # Format used: {grid: distance}}
        for comb in itertools.combinations(range(self.m * self.n), self.k):
            comb = list(comb)
            src = comb + [i + 1 for i in comb]
            # Format used: (i1, i2, ..., ik, j1, j2, ..., jk)
            # where (j1, j2, ..., jk) are the numbers of the tiles
            # considered and (i1, i2, ..., ik) their respective
            # positions
            src = tuple(src)
            queue = collections.deque([(src, 0)])
            seen = set([src])
            while queue:
                node, d = queue.pop()
                half_m = half_manhattan_distance(self.k, self.n, node)
                if d > half_m:
                    self.gadb[node] = d - half_m
                    # We only store the difference between the two distances
                for neighbor in get_neighbors(self.m, self.n, node):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.appendleft((neighbor, d + 1))

    @staticmethod
    def get_filename(m: int, n: int, k: int) -> str:
        """
        Gives a generic filename for storing APDBs.
        """
        return f"gadb\\{m} x {n}, {k}"

    def save(self, filename="") -> None:
        if not self.gadb:
            self.compute
        filename = filename or GADB.get_filename(self.m, self.n, self.k)
        with open(filename, "wb") as f:
            f.write(gzip.compress(pickle.dumps(self)))

    @classmethod
    def load(cls, filename) -> "GADB":
        with open(filename, "rb") as f:
            return pickle.loads(gzip.decompress(f.read()))

    @classmethod
    def default_load(cls, m: int, n: int, k: int) -> "GADB":
        return GADB.load(GADB.get_filename(m, n, k))


class GADBList:

    def __init__(self, gabds: List["GADB"]) -> None:
        self.gadbs = gabds

    def heuristic(self, grid: Tuple[int, ...]) -> float:
        """
        Returns an heuristic using the GADBs.
        """
        m, n = grid[0], grid[1]
        graph = []
        for gadb in self.gadbs:
            for comb in itertools.combinations(range(m * n), gadb.k):
                comb = [*comb]
                el = tuple(comb + [grid[i + 2] for i in comb])
                if el in gadb.gadb:
                    graph.append((gadb.gadb[el], gadb.k, el))
        # Solving the maximum weight matching in an hypergraph using a greedy algorithm
        graph.sort(key=lambda j: j[0] / j[1], reverse=True)
        weight = 0
        vertices = set()
        i = 0
        while len(vertices) < m * n and i < len(graph):
            w, k, el = graph[i]
            new_vertices = set(el[k:])
            if vertices.isdisjoint(new_vertices):
                vertices |= new_vertices
                weight += w
            i += 1
        return weight / 2 + utils.half_manhattan_distance(grid)

    def get_heuristic(self) -> Callable[[Tuple[int, ...]], float]:
        return lambda grid: self.heuristic(grid)
