# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid
import solver
import random
import itertools

solvers = [(solver.naive_solver, "Naive solver"), (solver.naive_bfs_solver, "Naive BFS solver"),
            (solver.bfs_solver, "BFS Solver"), (solver.manhattan_a_star_solver, "A* Solver with Manhattan heuristic")]
exact_solvers = solvers[1:]

def sample_random_permutations(m, n, k):
    samples = []
    perm = [*range(1, m * n + 1)]
    for i in range(k):
        cpy = perm.copy()
        random.shuffle(cpy)
        samples.append(Grid.grid_from_tuple([m, n] + cpy))
    return samples

class Test_Sort(unittest.TestCase):
    
    def test_is_sorted_solver(self):
        m, n = 3, 2
        grids = sample_random_permutations(m, n, 10)
        for solver in solvers:
            print(solver[1])
            for grid in grids:
                swaps = solver[0](grid)
                grid.swap_seq(swaps)
                self.assertEqual(grid.is_sorted(), True)
    
    def test_is_optimal_solver(self):
        m, n = 2, 3
        grids = sample_random_permutations(m, n, 10)
        length1, length2 = -1, -1
        for grid in grids:
            for i, solver in enumerate(exact_solvers):
                length2 = length1
                length1 = len(solver[0](grid))
                if i > 0:
                    print(exact_solvers[i-1][1], "compared to", exact_solvers[i][1])
                    self.assertEqual(length1, length2)

    
if __name__ == "__main__":
    unittest.main()