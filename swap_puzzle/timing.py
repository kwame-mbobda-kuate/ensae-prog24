import time
from grid import Grid
import utils
import wd
import apdb
from apdb import ArrayAPDB
import gadb
from gadb import GADB
import matplotlib.pyplot as plt
import solver
import numpy as np
import os
import pickle


def save_grids(m, n, N, filepath):
    grids = [Grid.random_grid(m, n) for _ in range(N)]
    with open(filepath, "wb") as f:
        pickle.dump(grids, f)


def load_grids(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def build_heuristic_grids():
    if not os.path.exists("grids"):
        os.makedirs("grids")
    save_grids(3, 3, 10**4, "grids\\grid 10e4 3 x 3")
    save_grids(4, 4, 10**4, "grids\\grid 10e4 4 x 4")


def space_and_time_gadb(m, n, k):
    g = gadb.GADB(m, n, k)
    t = time.perf_counter()
    g.compute()
    dt = time.perf_counter() - t
    ram_space = g.gadb.nbytes / 1000
    g.save()
    disk_space = os.path.getsize(gadb.GADB.get_filename(m, n, k)) / 1000
    print(
        f"Computation of GADB {(m, n, k)} in {dt}s. It takes {ram_space}kB in the RAM and {disk_space}kB on the disk."
    )


def space_and_time_apdb(m, n, group):
    ap = apdb.ArrayAPDB(m, n, group)
    t = time.perf_counter()
    ap.compute()
    dt = time.perf_counter() - t
    ram_space = ap.apdb.nbytes / 1000
    ap.save()
    disk_space = os.path.getsize(apdb.ArrayAPDB.default_filename(m, n, group)) / 1000
    print(
        f"Computation of APDB {(m, n, group)} in {dt}s. It takes {ram_space}kB in the RAM and {disk_space}kB on the disk."
    )


def benchmark_heuristic(heuristic, name, grids):
    times = []
    values = []
    for grid in grids:
        t = time.perf_counter()
        value = heuristic(grid.to_tuple())
        dt = time.perf_counter() - t
        times.append(dt)
        values.append(value)
    time_mean, time_std = np.mean(times), np.std(times)
    value_mean, value_std = np.mean(values), np.std(values)
    print(f"{name}: {time_mean}s (±{time_std}), {value_mean} (± {value_std})")
    return values


def space_and_time_db():
    [space_and_time_gadb(3, 3, k) for k in range(1, 5)]
    [space_and_time_gadb(4, 4, k) for k in range(1, 4)]
    space_and_time_apdb(3, 3, (1, 2, 3, 4, 5, 6, 7, 8, 9))
    space_and_time_apdb(4, 4, (1, 2, 3, 5, 6))
    space_and_time_apdb(4, 4, (4, 7, 8, 11, 12))
    space_and_time_apdb(4, 4, (9, 10, 13, 14, 15, 16))
    space_and_time_gadb(4, 4, 4)
    # space_and_time_apdb(4, 4, (1, 2, 3, 4, 5, 6, 7)) Takes at least one hour !


def benchmark_all_heuristics():

    space_and_time_db()

    gadbs_3_3 = [gadb.GADB.default_load(3, 3, k) for k in range(1, 5)]
    gadbs_4_4 = [gadb.GADB.default_load(4, 4, k) for k in range(1, 5)]

    h_gadb_3_3_2 = gadb.GADBList(gadbs_3_3[:2]).heuristic
    h_gadb_3_3_2_frac = gadb.GADBList(gadbs_3_3[:2]).heuristic_frac
    h_gadb_3_3_4 = gadb.GADBList(gadbs_3_3).heuristic
    h_gadb1_3_3_4_frac = gadb.GADBList(gadbs_3_3).heuristic_frac

    h_gadb_4_4_2 = gadb.GADBList(gadbs_4_4[:2]).heuristic
    h_gadb_4_4_2_frac = gadb.GADBList(gadbs_4_4[:2]).heuristic_frac
    h_gadb_4_4_4 = gadb.GADBList(gadbs_4_4).heuristic
    h_gadb1_4_4_4_frac = gadb.GADBList(gadbs_4_4).heuristic_frac

    ap_3_3_1 = ArrayAPDB.default_load(3, 3, (1, 2, 3, 4, 5, 6, 7, 8, 9))
    ap_4_4_1 = ArrayAPDB.default_load(4, 4, (1, 2, 3, 5, 6))
    ap_4_4_2 = ArrayAPDB.default_load(4, 4, (4, 7, 8, 11, 12))
    ap_4_4_3 = ArrayAPDB.default_load(4, 4, (9, 10, 13, 14, 15, 16))
    # ap_4_4_4 = ArrayAPDB.default_load(4, 4, (1, 2, 3, 4, 5, 6, 7))
    # ap_4_4_5 = apdb.VertSymmetryAPDB(ap_4_4_4)
    # ap_4_4_6 = ArrayAPDB.default_load(4, 4, (11, 12))

    h_apdb_1 = apdb.APDBList(lambda x: x, [ap_3_3_1]).heuristic
    h_apdb_2 = apdb.APDBList(utils.half_sum, [ap_4_4_1, ap_4_4_2, ap_4_4_3]).heuristic
    # h_apdb_3 = apdb.APDBList(utils.half_sum, [ap_4_4_4, ap_4_4_5, ap_4_4_6]).heuristic

    wd_3_3 = wd.WDDB(3, 3)
    wd_3_3.compute()
    h_wd_3_3 = wd_3_3.heuristic
    wd_4_4 = wd.WDDB(4, 4)
    wd_4_4.compute()
    h_wd_4_4 = wd_4_4.heuristic

    grids_3_3 = load_grids("grids\\grid 3 x 3")
    grids_4_4 = load_grids("grids\\grid 4 x 4")

    heuristics_3_3 = [
        utils.integer_half_manhattan_distance,
        utils.inversion_distance,
        h_wd_3_3,
        h_apdb_1,
        h_gadb_3_3_2,
        h_gadb_3_3_2_frac,
        h_gadb_3_3_4,
        h_gadb1_3_3_4_frac,
    ]
    names_3_3 = [
        "DM",
        "DI",
        "WD",
        "APDB 3 x 3 (1 2 3 4 5 6 7 8 9)",
        "GADB 3 x 3 2",
        "GADB 3 x 3 2 frac",
        "GADB 3 x 3 4",
        "GADB 3 x 3 4 frac",
    ]
    heuristics_values = [
        benchmark_heuristic(heuristic, name, grids_3_3)
        for heuristic, name in zip(heuristics_3_3, names_3_3)
    ]
    for i, heuristic_values in enumerate(heuristics_values):
        plt.plot([*range(50)], heuristic_values[:50], label=names_3_3[i])
    plt.legend(title="Heuristique")
    plt.title("Valeurs des heuristiques sur les 50 premières grilles (3 x 3)")
    plt.show()

    heuristics_4_4 = [
        utils.integer_half_manhattan_distance,
        utils.inversion_distance,
        h_wd_4_4,
        h_apdb_2,
        # h_apdb_3,
        h_gadb_4_4_2,
        h_gadb_4_4_2_frac,
        h_gadb_4_4_4,
        h_gadb1_4_4_4_frac,
    ]
    names_4_4 = [
        "DM",
        "DI",
        "WD",
        "APDB 4 x 4 (1, 2, 3, 5, 6) + (4, 7, 8, 11, 12) + (9, 10, 13, 14, 15, 16)",
        # "APDB 4 x 4 (1, 2, 3, 4, 5, 6, 7) + (8, 9, 10, 13, 14, 15, 16) + (11, 12)",
        "GADB 4 x 4 2",
        "GADB 4 x 4 2 frac",
        "GADB 4 x 4 4",
        "GADB 4 x 4 4 frac",
    ]
    heuristics_values = [
        benchmark_heuristic(heuristic, name, grids_4_4)
        for heuristic, name in zip(heuristics_4_4, names_4_4)
    ]
    for i, heuristic_values in enumerate(heuristics_values):
        plt.plot([*range(50)], heuristic_values[:50], label=names_4_4[i])
    plt.legend(title="Heuristique")
    plt.title("Valeurs des heuristiques sur les 50 premières grilles (4 x 4)")
    plt.show()


def build_solver_grids():
    if not os.path.exists("grids"):
        os.makedirs("grids")
    save_grids(3, 3, 100, "grids\\grid 100 3 x 3")
    save_grids(4, 3, 50, "grids\\grid 50 4 x 3")
    save_grids(4, 4, 5, "grids\\grid 5 4 x 4")


def benchmark_solver(solver, grids):
    times = []
    paths_len = []
    nb_nodes = []
    for grid in grids:
        t = time.perf_counter()
        dict_ = solver.solve(grid, debug=True)
        dt = time.perf_counter() - t
        times.append(dt)
        nb_nodes.append(dict_["nb_nodes"])
        paths_len.append(len(dict_["path"]))
    time_mean, time_std = np.mean(times), np.std(times)
    nodes_mean, nodes_std = np.mean(nb_nodes), np.std(nb_nodes)
    path_mean, path_std = np.mean(paths_len), np.std(paths_len)
    print(
        f"{solver}: {time_mean}s (±{time_std}), {path_mean} swaps (± {path_std}), {nodes_mean} nodes (± {nodes_std})"
    )
    return nb_nodes, paths_len


def benchmark_all_solvers():

    grids_3_3 = load_grids("grids\\grid 100 3 x 3")
    grids_4_3 = load_grids("grids\\grid 50 4 x 3")
    grids_4_4 = load_grids("grids\\grid 5 4 x 4")

    bfs_solvers = [
        solver.BFSSolver("BFS Solver"),
        solver.BidirectionalBFSSolver("Bidirectionnal BFS"),
        solver.PseudoBidirectionalSolver("Pseudo Bidirectionnal BFS Solver"),
    ]
    a_star_solvers = [
        solver.NoTieBreakAStarSolver(
            utils.half_manhattan_distance, "A* without TB and with MD"
        ),
        solver.NoTieBreakBucketAStarSolver(
            utils.integer_half_manhattan_distance,
            "A* without TB and with MD and bucket list",
        ),
        solver.AStarSolver(utils.half_manhattan_distance, "A* Solver with MD"),
        solver.AStarSolver(
            utils.integer_half_manhattan_distance, "A* solver with integer MD"
        ),
        solver.BucketAStarSolver(
            utils.integer_half_manhattan_distance, "A* with MD and bucket list"
        ),
        solver.MDAStarSolver("A* Solver with incremental computation of MD"),
        solver.AStarSolver(
            lambda g: 3 * utils.half_manhattan_distance(g), "WA* Solver with MD (w = 3)"
        ),
    ]
    bid_solvers = [
        solver.PerimeterAStarSolver(
            utils.alt_general_half_manhattan_distance, 1, "PA*_1 with MD"
        ),
        solver.BidirectionalAStarSolver(
            utils.general_half_manhattan_distance,
            utils.general_half_manhattan_distance,
            "BA*",
        ),
        solver.DIBBSSolver(
            utils.general_half_manhattan_distance,
            utils.general_half_manhattan_distance,
            "DIBBS with MD",
        ),
    ]

    wd_3_3 = wd.WDDB(3, 3)
    wd_3_3.compute()
    h_wd_3_3 = wd_3_3.heuristic
    s_wd_3_3 = solver.AStarSolver(h_wd_3_3, "A* with WD")
    wd_4_3 = wd.WDDB(4, 3)
    wd_4_3.compute()
    h_wd_4_3 = wd_4_3.heuristic
    s_wd_4_3 = solver.AStarSolver(h_wd_4_3, "A* with WD")
    wd_4_4 = wd.WDDB(4, 4)
    wd_4_4.compute()
    h_wd_4_4 = wd_4_4.heuristic
    s_wd_4_4 = solver.AStarSolver(h_wd_4_4, "A* with WD")

    print("Benchmarking of 3x3 solvers")
    [
        benchmark_solver(solver, grids_3_3)
        for solver in bfs_solvers + a_star_solvers + [s_wd_3_3] + bid_solvers
    ]
    print()
    print("Benchmarking of 4x3 solvers")
    [
        benchmark_solver(solver, grids_4_3)
        for solver in a_star_solvers + [s_wd_4_3] + bid_solvers
    ]
    print()
    print("Benchmarking of 4x4 solvers")
    [benchmark_solver(solver, grids_4_4) for solver in a_star_solvers + [s_wd_4_4]]


if __name__ == "__main__":
    benchmark_all_heuristics()
    benchmark_all_solvers()
