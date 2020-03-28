from src.NumericalIntegration import trial_solution
from src.Units import R_sun
from src.Graphing import graph_star
from src.ExampleStar import load_example_data

if __name__ == '__main__':
    r_values, state_values, error = trial_solution(58.556e3, 8.23544e+06, R_sun / 10 ** 5)
    reference_data = load_example_data()
    graph_star(r_values, state_values, "Test_hardcode", reference_data=reference_data)
