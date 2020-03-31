import numpy as np
from NumericalIntegration import solve_numerically
from Units import R_sun, million_K
from Graphing import graph_star
from HR_Plot import get_luminosity, hr_plot, generate_hr
from ExampleStar import load_example_data


if __name__ == '__main__':
    reference_data = load_example_data()
    # r_values, state_values, error = trial_solution_rk45(58.56399787e3, 8.23544e+06, R_sun / 10 ** 5, return_star=True)
    # graph_star(r_values, state_values, "Test_hardcode_rk45", reference_data=load_example_data)

    generate_hr()
