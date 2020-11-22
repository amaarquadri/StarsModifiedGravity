import numpy as np
from scipy import stats
from src.units import example_star_units
from src.stellar_structure_equations import StellarConfiguration, rho_index, T_index, M_index, L_index

_, ex_r_index, \
    ex_rho_index, ex_T_index, ex_M_index, ex_L_index, \
    ex_L_prime_index, ex_L_proton_proton_prime_index, ex_L_CNO_prime_index, \
    ex_dlog_P_by_dlog_T_index, ex_r_fraction_index, \
    _, _, _, _, \
    ex_kappa_index, ex_kappa_H_minus_index, ex_kappa_ff_index, ex_kappa_es_index, \
    ex_P_index, ex_P_degeneracy_index, ex_P_gas_index, ex_P_photon_index, \
    ex_rho_prime_index, ex_T_prime_index, ex_M_prime_index, _ \
    = np.arange(27)


def test_stellar_structure_equations(file_name="../Example Stars/low_mass_star.txt", config=StellarConfiguration()):
    data = np.loadtxt(file_name).T * example_star_units[:, None]

    diff = config.T_prime_radiative(data[ex_r_index, :], data[ex_rho_index, :], data[ex_T_index, :],
                                    data[ex_L_index, :]) - \
        config.T_prime_convective(data[ex_r_index, :], data[ex_rho_index, :], data[ex_T_index, :],
                                  data[ex_M_index, :])

    rho_prime_actual = config.rho_prime(data[ex_r_index, :], data[ex_rho_index, :], data[ex_T_index, :],
                                        data[ex_M_index, :], data[ex_L_index, :])
    rho_prime_expected = data[ex_rho_prime_index, :]
    print("Rho Prime Percentage Error:", stats.describe((rho_prime_actual - rho_prime_expected) / rho_prime_expected))

    T_prime_actual = config.T_prime(data[ex_r_index, :], data[ex_rho_index, :], data[ex_T_index, :],
                                    data[ex_M_index, :], data[ex_L_index, :])
    T_prime_expected = data[ex_T_prime_index, :]
    print("T Prime Percentage Error:", stats.describe((T_prime_actual - T_prime_expected) / T_prime_expected))

    M_prime_actual = config.M_prime(data[ex_r_index, :], data[ex_rho_index, :])
    M_prime_expected = data[ex_M_prime_index, :]
    print("M PrimePercentage Error:", stats.describe((M_prime_actual - M_prime_expected) / M_prime_expected))

    L_prime_actual = config.L_prime(data[ex_r_index, :], data[ex_rho_index, :], data[ex_T_index, :])
    L_prime_expected = data[ex_L_prime_index, :]
    print("L prime Percentage Error:", stats.describe((L_prime_actual - L_prime_expected) / L_prime_expected))

    P_actual = config.P(data[ex_rho_index, :], data[ex_T_index, :])
    P_expected = data[ex_P_index, :]
    print("P Percentage Error:", stats.describe((P_actual - P_expected) / P_expected))

    P_degeneracy_actual = config.P_degeneracy(data[ex_rho_index, :])
    P_degeneracy_expected = data[ex_P_degeneracy_index, :]
    print("P_degeneracy Percentage Error:",
          stats.describe((P_degeneracy_actual - P_degeneracy_expected) / P_degeneracy_expected))

    P_gas_actual = config.P_gas(data[ex_rho_index, :], data[ex_T_index, :])
    P_gas_expected = data[ex_P_gas_index, :]
    print("P_gas Percentage Error:", stats.describe((P_gas_actual - P_gas_expected) / P_gas_expected))

    kappa_actual = config.kappa(data[ex_rho_index, :], data[ex_T_index, :])
    kappa_expected = data[ex_kappa_index, :]
    print("Kappa Percentage Error:", stats.describe((kappa_actual - kappa_expected) / kappa_expected))


def load_example_data(file_name="../Example Stars/low_mass_star.txt"):
    return np.loadtxt(file_name).T * example_star_units[:, None]


def evaluate_star(r_values, state_matrix, reference_data):
    errors = []
    for val_index, ex_val_index in [(rho_index, ex_rho_index), (T_index, ex_T_index),
                                    (M_index, ex_M_index), (L_index, ex_L_index)]:
        errors.append(np.mean(np.abs(np.interp(reference_data[ex_r_index], r_values, state_matrix[val_index, :]) -
                                     reference_data[ex_val_index, :])))
    return np.sqrt(np.mean(np.array(errors) ** 2))


if __name__ == '__main__':
    test_stellar_structure_equations()
