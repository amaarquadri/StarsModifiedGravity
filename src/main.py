import numpy as np
from src.stellar_structure_equations import StellarConfiguration
from src.numerical_integration import solve_bvp
from src.units import R_sun, million_K, g, cm
from src.graphing import graph_star
from src.example_star import load_example_data
from src.star_sequence_generator import generate_stars


if __name__ == '__main__':
    reference_data = load_example_data(file_name='../Example Stars/high_mass_star.txt')
    config = StellarConfiguration()
    error, r_values, state_values = solve_bvp(20 * million_K,
                                              rho_c_guess=80.063 * g / cm ** 3,
                                              confidence=0.5,
                                              config=config)
    print('Error', error)
    graph_star(r_values, state_values, 'high_mass', config=config, reference_data=reference_data)
