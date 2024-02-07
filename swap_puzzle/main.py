import grid
import graphics
import solver

n, m = 4, 4
grid = grid.Grid.random_grid(m, n)
print(grid)
print(solver.manhattan_a_star_solver(grid))
graphics.game(grid, 800, 800)