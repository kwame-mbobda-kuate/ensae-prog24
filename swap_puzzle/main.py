import grid
import graphics
import solver
import utils
import apdb

m, n = 3, 3
groups = [[1, 2, 3, 5, 6], [9, 10, 13, 14, 15, 16], [4, 7, 8, 11, 12]]
# h = apdb.apdb_heuristic(m, n, groups)
# s1 = solver.AStarSolver(h, "A* with APDB")
s2 = solver.AStarSolver(utils.half_manhattan_distance, "A* with MD")
s3 = solver.AStarSolver(utils.linear_conflict, "A* with LC")
g = grid.Grid.random_grid(m, n)
while len(s3.solve(g)) == len(s2.solve(g)):
    g = grid.Grid.random_grid(m, n)
print(s3, len(s3.solve(g)))
print(s2, len(s2.solve(g)))
