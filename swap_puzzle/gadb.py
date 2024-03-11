from grid import Grid
import itertools
import pickle
import pickle
import gzip
import numpy as np
import utils
import math
from typing import List, Tuple, Set
import collections


def manhattan_distance(
    n: int, coords: Tuple[int, ...], numbers: Tuple[int, ...]
) -> int:
    """
    Compute the half sum of the Manhattan distances between the cell occupied by each number
    in a given grid and the cell it should occupy in the sorted grid.
    The representation used here is different from the usual one.
    """
    return (
        sum(
            abs(((numbers[i] - 1) % n - (coords[i] % n)))
            + abs((numbers[i] - 1) // n - coords[i] // n)
            for i in range(len(coords))
        )
        / 2
    )


def get_neighbors(m: int, n: int, coords: Tuple[int, ...]) -> Set[Tuple[int, ...]]:
    """
    Determine the grids reachable in a swap from a given grid.
    The representation used here is different from the usual one.
    """
    neighbors = []
    N = len(coords)
    #We start with swaps between two numbers
    for i in range(N):
        for j in range(i):
            if (
                abs(coords[i] - coords[j]) == 1 and coords[i] // n == coords[j] // n
            ) or abs(coords[i] - coords[j]) == n:
                L = list(coords)
                L[i], L[j] = L[j], L[i]
                neighbors.append(tuple(L))
    #We consider swap between involving a single number
    for i in range(N):
        for j in (-1, 1):
            if (
                coords[i] // n == (coords[i] + j) // n
                and 0 <= coords[i] + j < m * n
                and coords[i] + j not in coords
            ):
                L = list(coords)
                L[i] += j
                neighbors.append(tuple(L))
            if 0 <= coords[i] + j * n < m * n and coords[i] + j * n not in coords:
                L = list(coords)
                L[i] += j * n
                neighbors.append(tuple(L))
    return set(neighbors)


class GADB:
    """
    Represents a General Additive Database (or also called dynamically-partitionned database)
    as described in https://arxiv.org/pdf/1107.0050.pdf (section 3).
    """

    def __init__(self, m: int, n: int, k: int, gadb: "np.ndarray" = []):
        self.m = m
        self.n = n
        self.k = k
        self.ranking_combinations = dict() #bijection from the set of combinations to integers
        self.ranking_permutations = dict() #bijection from the set of permutations to integers
        self.gadb = gadb

    def compute(self):
        """
        This functions finds all the k-uplets of tiles whose distance is greater than
        their half Mahnattan distance.
        """
        perm_rank = 0
        neighbors = {}  # Precomputation of the neighbors
        for perm in itertools.permutations(range(self.m * self.n), self.k):
            self.ranking_permutations[perm] = perm_rank
            neighbors[perm] = get_neighbors(self.m, self.n, perm)
            perm_rank += 1
        self.gadb = np.zeros(
            [
                math.comb(self.m * self.n, self.k),
                math.factorial(self.n * self.m)
                // math.factorial(self.n * self.m - self.k),
            ],
            dtype=np.int8,
        )
        comb_rank = 0
        nb = 0
        for numbers in itertools.combinations(range(1, self.m * self.n + 1), self.k):
            self.ranking_combinations[numbers] = comb_rank
            src = tuple(i - 1 for i in numbers)
            # Format used: ((i1, i2, ..., ik), (j1, j2, ..., jk))
            # where (j1, j2, ..., jk) are the numbers of the tiles
            # considered and (i1, i2, ..., ik) their respective
            # coordinates
            queue = collections.deque([(src, 0)])
            seen = set([src])
            while queue:
                coords, d = queue.pop()
                permutation_rank = self.ranking_permutations[coords]
                self.gadb[comb_rank][permutation_rank] = d
                nb += 1
                for neighbor in neighbors[coords]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.appendleft((neighbor, d + 1))
            comb_rank += 1

    @staticmethod
    def get_filename(m: int, n: int, k: int) -> str:
        """
        Gives a generic filename for storing GADBs.
        """
        return f"gadb\\{m} x {n}, {k}"

    def save(self, filename="") -> None:
        if self.gadb == []:
            self.compute()
        filename = filename or GADB.get_filename(self.m, self.n, self.k)
        with gzip.open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename) -> "GADB":
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)

    @classmethod
    def default_load(cls, m: int, n: int, k: int) -> "GADB":
        return GADB.load(GADB.get_filename(m, n, k))

    def weight(self, coords, numbers) -> float:
        return (
            self.gadb[self.ranking_combinations[numbers]][
                self.ranking_permutations[coords]
            ]
            
        )


class GADBList:

    def __init__(self, gabds: List["GADB"]) -> None:
        self.gadbs = gabds

    @classmethod
    def batch_load(cls, m: int, n: int, k: int):
        gadb_list = []
        for j in range(2, k + 1):
            gadb_list.append(GADB.default_load(m, n, j))
        return GADBList(gadb_list)

    def hypergraph(self, grid):
        m, n = grid[0], grid[1]
        hypergraph = []
        inv = utils.inverse(grid)
        for gadb in self.gadbs:
            for comb in itertools.combinations(range(1, m * n + 1), gadb.k):
                coords = tuple(inv[i + 1] - 1 for i in comb)
                w = gadb.weight(coords, comb)
                hypergraph.append((w, gadb.k, coords, comb))
        return hypergraph

    def heuristic(self, grid: Tuple[int, ...]) -> float:
        """
        Returns an heuristic using the GADBs.
        """
        m, n = grid[0], grid[1]
        hypergraph = self.hypergraph(grid)
        # Solving the maximum weight matching in an hypergraph using a greedy algorithm
        hypergraph.sort(reverse=True)
        weight = 0
        vertices = set()
        i = 0
        while len(vertices) < m * n and i < len(hypergraph):
            w, k, coords, numbers = hypergraph[i]
            new_vertices = set(numbers)
            if vertices.isdisjoint(new_vertices):
                vertices |= new_vertices
                weight += w
            i += 1
        return math.ceil(weight / 2)
    
    def heuristic_frac(self, grid: Tuple[int, ...]) -> float:
        """
        Returns an heuristic using the GADBs.
        """
        m, n = grid[0], grid[1]
        hypergraph = self.hypergraph(grid)
        # Solving the maximum weight matching in an hypergraph using a greedy algorithm
        hypergraph.sort(key=lambda j: j[0] / j[1], reverse=True)
        weight = 0
        vertices = set()
        i = 0
        while len(vertices) < m * n and i < len(hypergraph):
            w, k, coords, numbers = hypergraph[i]
            new_vertices = set(numbers)
            if vertices.isdisjoint(new_vertices):
                vertices |= new_vertices
                weight += w
            i += 1
        return math.ceil(weight / 2)