from grid import Grid
from solver import Solver

g = Grid(2, 2)
g.swap((1, 1), (0, 1))
print(g)
s = Solver()
print(s.bfs_solve(g))
print(g)
#graphics.game(g, 500, 500)

"""g = Grid(2, 3)
S = Solver()
show(g, True)
print(g)
print(S.get_solution(g))
print(g)


data_path = "input/"
file_name = data_path + "grid0.in"

print(file_name)

g = Grid.grid_from_file(file_name)
print(g)
swaps = S.get_solution(g)
print(swaps)
print(g)"""

