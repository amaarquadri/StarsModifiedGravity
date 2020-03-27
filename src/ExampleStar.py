import numpy as np
from scipy import stats
from src.Units import example_star_units
from src.StellarStructureEquations import rho_prime

_, ex_r_index, \
    ex_rho_index, ex_T_index, ex_M_index, ex_L_index, \
    ex_L_prime_index, ex_L_proton_proton_prime_index, ex_L_CNO_prime_index, \
    ex_dlog_P_by_dlog_T_index, ex_r_fraction_index, \
    _, _, _, _, \
    ex_kappa_index, ex_kappa_H_minus_index, ex_kappa_ff_index, ex_kappa_es_index, \
    ex_P_index, ex_P_degeneracy_index, ex_P_gas_index, ex_P_photon_index, \
    ex_rho_prime_index, ex_T_prime_index, ex_M_prime_index, _ \
    = np.arange(27)


def test_stellar_structure_equation(file_name="../Example Stars/lowmass_star.txt"):
    data = np.loadtxt(file_name).T * example_star_units[:, None]

    actual = rho_prime(data[ex_r_index, :], data[ex_rho_index, :], data[ex_T_index, :], data[ex_M_index, :],
                       data[ex_L_index, :])
    expected = data[ex_rho_prime_index, :]
    print("Ratio Error:", stats.describe(actual / expected))
    print("Percentage Error:", stats.describe((actual - expected) / expected))


def load_example_data(file_name="../Example Stars/lowmass_star.txt"):
    return np.loadtxt(file_name).T * example_star_units[:, None]


if __name__ == '__main__':
    test_stellar_structure_equation()
