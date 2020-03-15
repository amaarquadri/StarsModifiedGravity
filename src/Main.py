from src.NumericalIntegration import *
from src.Graphing import graph_star

if __name__ == '__main__':
    r_values, state_values, error = trial_solution_rk45(160e3, 15.7e6, 10_000)
    graph_star(r_values, state_values)
