import solver
import utils
import grid
import graphics

g = utils.difficulty_bounded_grid(4, 4, 0.5, 0.7)
graphics.game(g, 500, 500)
print(solver.MDAStarSolver().solve(g))