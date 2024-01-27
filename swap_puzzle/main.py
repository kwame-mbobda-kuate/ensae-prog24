import random
import grid
import graphics
import solver

S = solver.Solver()
n, m = 4, 4
state = [*range(1, 1 + n * m)]
random.shuffle(state)
grid = grid.Grid.grid_from_tuple_2(m, n, state)
graphics.game(grid, 600, 600)
