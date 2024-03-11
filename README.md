# ensae-prog24
Repository of Quentin MALLEGOL and Kwame MBOBDA-KUATE for the ENSAE 1A programming project 2023-24 on the swap puzzle. 

│   .gitignore
│   README.md
│       
├───img # Images for the swap puzzle (unused)
│       blue_tile.PNG
│       green_tile.PNG
│       red_tile.PNG
│       
├───input # Test inputs and outputs
│       graph1.in
│       graph1.path.out
│       graph2.in
│       graph2.path.out
│       grid0.in
│       grid1.in
│       grid2.in
│       grid3.in
│       grid4.in
│
├───swap_puzzle # Main files of the project
│       apdb.py # Additive pattern databse aka pattern database statically computed
│       benchmark.py # Benchmarks heuristics and solvers
│       gadb.py # General additive databse aka pattern database dynamically computed
│       graph.py # Implementation of a graph with an adjency list
│       graphics.py # User interface with Pygame
│       grid.py # Representation a grid of the swap puzzle
│       main.py # File from which other functions are usually called. Contains a minimal code.
│       solver.py # Implementation of all the solvers
│       solver_utils.py # Miscellanous data structures needed by solvers
│       timing.py # Benchmarks all heuristic and solvers and show the result through matplotlib and console prints
│       utils.py # Miscellanous functions
│       wd.py # Walking Distance
│
└───tests # Tests of functions defined above
        test_allowed_swaps.py 
        test_graph.py
        test_grid_from_file.py
        test_heuristic.py
        test_is_sorted.py
        test_solver.py
        test_swap.py
        test_to_tuple.py
