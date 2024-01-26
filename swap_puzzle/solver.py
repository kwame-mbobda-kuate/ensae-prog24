from grid import Grid
import random
from graph import Graph
import itertools
import copy
import collections

def sign(x):
    if x > 0:
        return 1
    return -1

class Solver(): 
    """
    A solver class, to be implemented.
    """
    
    def get_solution(self, grid):
        """
        Solves the grid and returns the sequence of swaps at the format 
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...]. 
        """
        
        raise NotImplementedError
        

    def random_solve(self, grid):
        swaps = []
        while not grid.is_sorted():
            i1, j1 = random.randrange(grid.n), random.randrange(grid.m)
            i2, j2 = random.choice(grid.allowed_swaps((i1, j1)))
            swaps.append(((i1, j1), (i2, j2)))
        return swaps
    
    def naive_solve(self, grid):
        swaps = []
        for i in range(grid.n):
            for j in range(grid.m):
                k = (i+1)*(j+1)
                for i1 in range(grid.n):
                    for j1 in range(grid.m):
                        if grid.state[i1][j1] == k:
                            break
                    if grid.state[i1][j1] == k:
                            break
                while j1 != j:
                    swaps.apend((i1, j1), (i1, j1 + sign(j - j1)))
                    j1 += sign(j - j1)
                while i1 != i:
                    swaps.apend((i1, j1), (i1 + sign(i - i1), j1))
                    i1 += sign(i - i1)
        return swaps
    
    def naive_bfs_solve(self, grid):
        src = grid
        m, n = grid.m, grid.n
        nodes = []
        all_swaps = grid.all_swaps()
        for perm in itertools.permutations(range(1, 1 + m * n)):
            perm = [*perm]
            grid = Grid(m, n)
            for i in range(m):
                for j in range(n):
                    grid.state[i][j] = perm[i * n + j]
            nodes.append(grid)
        graph = Graph([grid.to_tuple() for grid in nodes])
        for grid in nodes:
            for swap in all_swaps:
                cpy = copy.deepcopy(grid)
                grid.swap(swap[0], swap[1])
                graph.add_edge(grid.to_tuple(), cpy.to_tuple())
        path = graph.bfs(src.to_tuple(), tuple([m, n] + [x + 1 for x in range(n * m)]))
        swap_sol = []
        for i in range(len(path) - 1):
            j1 = 0
            while path[i][j1 + 2] == path[i+1][j1 + 2]:
                j1 += 1
            #Petite arnaque (on teste un échange qui n'est pas forcément valide)
            if j1 > 0 and path[i][j1 + 1] != path[i+1][j1 + 1]:
                j2 = j1 - 1
            elif j1 < n*m - 1 and  path[i][j1 + 3] != path[i+1][j1 + 3]:
                j2 = j1 + 1
            elif j1 < n*m - grid.n and path[i][j1 + 2 + grid.n] != path[i + 1][j1 + 2 + grid.n]:
                j2 = j1 + grid.n
            else:
                j2 = j1 - grid.n
            swap_sol.append(((j1 // grid.n, j1 % grid.n), (j2 // grid.n , j2 % grid.n)))
        return swap_sol

    def bfs_solve(self, grid):
        all_swaps = grid.all_swaps()
        m, n = grid.m, grid.n
        src = grid.to_tuple()
        dst = Grid(m, n).to_tuple()
        seen = set([src])
        queue = collections.deque([(src, [])])
        node = None
        while queue and (node is None or node[0] != dst):
            node = queue.pop()
            L = list(node[0])
            for swap in all_swaps:
                swap_arr(L, swap[0][0]*n + swap[0][1] + 2, swap[1][0]*n + swap[1][1] + 2)
                cpy = tuple(L)
                if cpy not in seen:
                    path = node[1] + [swap]
                    queue.appendleft((cpy, path))
                    seen.add(cpy)
                swap_arr(L, swap[0][0]*n + swap[0][1] + 2, swap[1][0]*n + swap[1][1] + 2)
        return node[1]


def swap_arr(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]