# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid
from graph import Graph


class Test_Graph(unittest.TestCase):
    def aux_test_graph(self, input_filename, output_filename):
        graph = Graph.graph_from_file(input_filename)
        with open(output_filename, "r") as f:
            for line in f.readlines():
                l = line.split()
                src = int(l[0])
                dst = int(l[1])
                if "None" in line:
                    path = "None"
                else:
                    path = line[line.index("[") :].strip()
                print(path)
                self.assertEqual(path, str(graph.bfs(src, dst)))

    def test_graph1(self):
        self.aux_test_graph("input/graph1.in", "input/graph1.path.out")

    def test_graph2(self):
        self.aux_test_graph("input/graph2.in", "input/graph2.path.out")


if __name__ == "__main__":
    unittest.main()
