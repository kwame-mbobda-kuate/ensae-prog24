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


def half_manhattan_distance(m: int, n: int, grid: Tuple[int, ...]) -> float:
    """
    Compute the half sum of the Manhattan distances between the cell occupied by each number
    in a given grid and the cell it should occupy in the sorted grid.
    The representation used here is different from the usual one.
    """
    N = len(grid) // 2
    return (
        sum(
            abs(((grid[i + N] - 1) % n - (grid[i] % n)))
            + abs((grid[i + N] - 1) // n - grid[i] // n)
            for i in range(N)
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


def compute_gadb(m: int, n: int, k: int):
    """
    Computes a General Additive Database (or also called dynamically-partitionned database)
    as described in https://arxiv.org/pdf/1107.0050.pdf. This functions finds all
    the k-uplets of tiles whose distance is superior than their half Mahnattan distance.
    """
    database = dict()
    # Format used: {grid: distance}
    for comb in itertools.combinations(range(m * n), k):
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
            if d > half_manhattan_distance(m, n, node):
                database[node] = d - half_manhattan_distance(m, n, node)
                # We only store the difference between the two distances
            for neighbor in get_neighbors(m, n, node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.appendleft((neighbor, d + 1))
    return database


def gadb_heuristic(m: int, n: int, gadbs):
    """
    Returns an heuristic using the GADBs.
    """

    def heuristic(grid):
        graph = []
        for db in gadbs:
            k, db = db
            for comb in itertools.combinations(range(m * n), k):
                comb = [*comb]
                el = tuple(comb + [grid[i + 2] for i in comb])
                if el in db:
                    graph.append((db[el], k, el))
        # Solving the maximum weight matching in an hypergraph using a greedy algorithm
        graph.sort(reverse=True)
        weight = 0
        vertices = []
        i = 0
        while len(vertices) < m * n and i < len(graph):
            w, k, el = graph[i]
            new_vertices = el[k:]
            if are_disjoint(vertices, new_vertices):
                vertices.extend(new_vertices)
                weight += w
            i += 1
        return weight + utils.half_manhattan_distance(grid)

    return heuristic


def get_filename(m: int, n: int, k: int) -> str:
    """
    Gives a generic filename for storing APDBs.
    """
    return f"gadb\\{m} x {n}, {k}"


def save_gadb(m: int, n: int, k: int, filename=""):
    db = compute_gadb(m, n, k)
    filename = filename or get_filename(m, n, k)
    with open(filename, "wb") as f:
        f.write(gzip.compress(pickle.dumps(db)))


def load_gadb(filename):
    with open(filename, "rb") as f:
        return pickle.loads(gzip.decompress(f.read()))