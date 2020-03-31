from src.NumericalIntegration import solve_numerically
from src.Units import R_sun
from src.Graphing import graph_star
from src.ExampleStar import load_example_data

if __name__ == '__main__':
    r_values, state_values = solve_numerically(8.23544e+06)
    # r_values, state_values, error = trial_solution_rk45(58.56399787e3, 8.23544e+06, R_sun / 10 ** 5, return_star=True)
    reference_data = load_example_data()
    graph_star(r_values, state_values, "Test_hardcode_rk45", reference_data=reference_data)
