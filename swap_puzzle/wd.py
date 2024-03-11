from typing import Dict, Tuple
import collections


class WDDB:
    """
    Represents a database for the Walking Distance and Inversion Distance.
    See https://web.archive.org/web/20141224035932/http://juropollo.xe0.ru:80/stp_wd_translation_en.htm
    for more information.
    """

    def __init__(self, m: int, n: int, db: Dict[Tuple[int, ...], int] = {}):
        self.m = m
        self.n = n
        self.db = db

    def compute(self):
        self.aux_compute(self.m, self.n)
        if self.n != self.m:
            self.aux_compute(self.n, self.m)

    def aux_compute(self, m, n):
        self.db[(m, n)] = {}
        src = [0] * m * m
        # Format: src[m * i + j]: how many tiles are in
        # line i and must go in line j, with
        # i and j both ranging from 0 to m-1
        for i in range(m):
            src[m * i + i] = n
        # BFS
        src = tuple(src)
        queue = collections.deque([src])
        self.db[(m, n)][src] = 0
        while queue:
            node = queue.pop()
            d = self.db[(m, n)][node]
            list_node = list(node)
            for i in range(m - 1):
                for j1 in range(m):
                    for j2 in range(m):
                        # if there's a tile in
                        # line i which and must go in line j1
                        # and another in line i which must go in line j1
                        if (
                            list_node[m * i + j1] > 0
                            and list_node[m * (i + 1) + j2] > 0
                        ):
                            # Then we swap them
                            list_node[m * i + j1] -= 1
                            list_node[m * (i + 1) + j1] += 1
                            list_node[m * (i + 1) + j2] -= 1
                            list_node[m * i + j2] += 1
                            neighbor = tuple(list_node)
                            list_node = list(node)
                            if neighbor not in self.db[(m, n)]:
                                self.db[(m, n)][neighbor] = d + 1
                                queue.appendleft(neighbor)

    def heuristic(self, grid: Tuple[int, ...]) -> int:
        grid_h = [0] * self.m**2
        grid_v = [0] * self.n**2

        for i in range(self.m):
            for j in range(self.n):
                line_final = (grid[self.n * i + j + 2] - 1) // self.n
                grid_h[self.m * i + line_final] += 1
        grid_h = tuple(grid_h)

        for i in range(self.n):
            for j in range(self.m):
                # Reflection of indices and values
                colum_final = (grid[self.n * j + i + 2] - 1) % self.n
                grid_v[self.n * i + colum_final] += 1
        grid_v = tuple(grid_v)

        wd_h = self.db[(self.m, self.n)][grid_h]
        wd_v = self.db[(self.n, self.m)][grid_v]

        return wd_h + wd_v
