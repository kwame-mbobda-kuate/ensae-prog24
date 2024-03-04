import itertools
from typing import Dict, List, Callable, Tuple
import collections

class WD:

    def __init__(self, m: int, n: int, db: Dict[Tuple[int, ...], int] = {}):
        self.m = m
        self.n = n
        self.db = db

    def compute(self):
        self.aux_compute(self.m, self.n)
        if self.n != self.m:
            self.aux_compute(self.n, self.m)
    
    def aux_compute(self, m, n):
        swaps = []
        self.db[(m, n)] = {}
        for i in range(m - 1):
            for j in range(n):
                for k in range(n):
                    swaps.append(((i, j), (i + 1, k)))
        src = tuple(tuple(n * i + j + 1 for j in range(n)) for i in range(m))
        queue = collections.deque([src])
        self.db[(m, n)][src] = 0
        while queue:
            node = queue.pop()
            d = self.db[(m, n)][node]
            list_node = [list(line) for line in node]
            for swap in swaps:
                (
                    list_node[swap[0][0]][swap[0][1]],
                    list_node[swap[1][0]][swap[1][1]],
                ) = (
                    list_node[swap[1][0]][swap[1][1]],
                    list_node[swap[0][0]][swap[0][1]],
                )
                list_node[swap[0][0]].sort()
                list_node[swap[1][0]].sort()
                neighbor = tuple(tuple(line) for line in list_node)
                (
                    list_node[swap[0][0]][swap[0][1]],
                    list_node[swap[1][0]][swap[1][1]],
                ) = (
                    list_node[swap[1][0]][swap[1][1]],
                    list_node[swap[0][0]][swap[0][1]],
                )
                if neighbor not in self.db[(m, n)]:
                    self.db[(m, n)][neighbor] = d + 1
                    queue.appendleft(neighbor)

    def heuristic(self, grid: Tuple[int, ...]):
        grid_h = tuple(
            tuple(sorted(grid[self.n * i + j + 2] for j in range(self.n)))
            for i in range(self.m)
        )
        grid_v = tuple(
            tuple(sorted(grid[self.n * i + j + 2] for i in range(self.m)))
            for j in range(self.n)
        )
        return self.db[(self.m, self.n)][grid_h] + self.db[(self.n, self.m)][grid_v]


import time
t = time.perf_counter()
WD(4, 4).compute()
print(time.perf_counter() - t)
