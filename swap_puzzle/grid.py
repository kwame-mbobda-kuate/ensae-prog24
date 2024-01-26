"""
This is the grid module. It contains the Grid class and its associated methods.
"""

class Grid():
    """
    A class representing the grid from the swap puzzle. It supports rectangular grids. 

    Attributes: 
    -----------
    m: int
        Number of lines in the grid
    n: int
        Number of columns in the grid
    state: list[list[int]]
        The state of the grid, a list of list such that state[i][j] is the number in the cell (i, j), i.e., in the i-th line and j-th column. 
        Note: lines are numbered 0..m and columns are numbered 0..n.
    """
    
    def __init__(self, m, n, initial_state = []):
        """
        Initializes the grid.

        Parameters: 
        -----------
        m: int
            Number of lines in the grid
        n: int
            Number of columns in the grid
        initial_state: list[list[int]]
            The intiail state of the grid. Default is empty (then the grid is created sorted).
        """
        self.m = m
        self.n = n
        if not initial_state:
            initial_state = [list(range(i*n+1, (i+1)*n+1)) for i in range(m)]            
        self.state = initial_state

    def __str__(self): 
        """
        Prints the state of the grid as text.
        """
        output = f"The grid is in the following state:\n"
        for i in range(self.m): 
            output += f"{self.state[i]}\n"
        return output

    def __repr__(self): 
        """
        Returns a representation of the grid with number of rows and columns.
        """
        return f"<grid.Grid: m={self.m}, n={self.n}>"

    def is_sorted(self):
        """
        Checks is the current state of the grid is sorte and returns the answer as a boolean.
        """
        k = 1
        for i in range(self.m):
            for j in range(self.n):
                if self.state[i][j] != k:
                    return False
                k += 1
        return True

    def swap(self, cell1, cell2):
        """
        Implements the swap operation between two cells. Raises an exception if the swap is not allowed.

        Parameters: 
        -----------
        cell1, cell2: tuple[int]
            The two cells to swap. They must be in the format (i, j) where i is the line and j the column number of the cell. 
        """
        if ((cell1[0] == cell2[0] and abs(cell1[1] - cell2[1]) == 1 and not (cell1[1] == self.n and cell2[1] == 0) and not (cell2[1] == 0 and cell2[1] == self.n)) or \
            (cell2[1] == cell2[1] and abs(cell1[0] - cell2[0]) == 1 and not (cell1[0] == self.m and cell2[0] == 0) and not (cell2[0] == 0 and cell2[0] == self.m))) and \
                0 <= cell1[0] < self.m and 0 <= cell1[1] < self.n and 0 <= cell2[0] < self.m and 0 <= cell2[1] < self.n:
            self.state[cell1[0]][cell1[1]], self.state[cell2[0]][cell2[1]] = self.state[cell2[0]][cell2[1]], self.state[cell1[0]][cell1[1]]
        else:
            raise Exception("Swap not allowed")

    def swap_seq(self, cell_pair_list):
        """
        Executes a sequence of swaps. 

        Parameters: 
        -----------
        cell_pair_list: list[tuple[tuple[int]]]
            List of swaps, each swap being a tuple of two cells (each cell being a tuple of integers). 
            So the format should be [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        for cell_pair in cell_pair_list:
            self.swap(*cell_pair)

    def allowed_swaps(self, cell):
        """
        Returns the swaps allowed from the cell given. 

        Parameters: 
        -----------
        cell: tuple[int]
            The cells from which the swaps will be done. It must be in the format (i, j) where i is the line and j the column number of the cell.
        
        Output:
        -------
        swaps: list[tuple[int]]
             The swaps allowed from the cell given. 
        """
        i, j = cell
        swaps = []
        if i > 0:
            swaps.append((i-1, j))
        if i < self.n - 1:
            swaps.append((i+1, j))
        if j > 0:
            swaps.append((i, j-1))
        if j < self.m - 1:
            swaps.append((i, j+1))
        return swaps

    @classmethod
    def grid_from_file(cls, file_name): 
        """
        Creates a grid object from class Grid, initialized with the information from the file file_name.
        
        Parameters: 
        -----------
        file_name: str
            Name of the file to load. The file must be of the format: 
            - first line contains "m n" 
            - next m lines contain n integers that represent the state of the corresponding cell

        Output: 
        -------
        grid: Grid
            The grid
        """
        with open(file_name, "r") as file:
            m, n = map(int, file.readline().split())
            initial_state = [[] for i_line in range(m)]
            for i_line in range(m):
                line_state = list(map(int, file.readline().split()))
                if len(line_state) != n: 
                    raise Exception("Format incorrect")
                initial_state[i_line] = line_state
            grid = Grid(m, n, initial_state)
        return grid

    
    def to_tuple(self):
        L = [self.m, self.n]
        for i in range(self.m):
            for j in range(self.n):
                L.append(self.state[i][j])
        return tuple(L)
    
    @classmethod
    def grid_from_tuple(tup):
        grid = Grid(tup[0], tup[1])
        for i in range(grid.m):
            for j in range(grid.n):
                grid.state[i][j] = tup[i * grid.n + j + 2]
        return grid

    def all_swaps(self):
        swaps = []
        #On n'ajoute que les échanges avec des cases situées à droite ou en bas
        for i in range(self.m):
            for j in range(self.n):
                if i < self.m - 1:
                    swaps.append([(i, j), (i+1, j)])
                if j < self.n - 1:
                    swaps.append([(i, j), (i, j + 1)])
        return swaps