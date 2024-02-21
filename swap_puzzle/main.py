import grid
import graphics
import solver
import utils
import apdb

m, n = 4, 4
groups = [[1, 2, 3, 5, 6], [9, 10, 13, 14, 15, 16], [4, 7, 8, 11, 12]]
h = apdb.apdb_heuristic(m, n, groups)
s1 = solver.AStarSolver(h, "A* with APDB")
s2 = solver.AStarSolver(utils.halved_manhattan_distance, "A* with MD")
g = grid.Grid.random_grid(m, n)
print(g)
print(s1.solve(g), s1)
print(s2.solve(g), s2)
