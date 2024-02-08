import grid
import graphics
import solver
import utils

n, m = 2, 2
grid = grid.Grid.random_grid(m, n)
print(grid)
s = solver.AStarSolver(utils.halved_manhattan_distance)
print(s.solve(grid))
graphics.game(grid, 600, 600)