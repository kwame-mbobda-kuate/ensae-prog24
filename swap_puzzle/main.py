import random
import grid
import graphics
import solver

n, m = 4, 4
grid = grid.Grid.random_grid(m, n)
print(solver.manhattan_a_star_solver(grid))
graphics.game(grid, 800, 800)
#graphics.game(grid, 800, 800)