import numpy as np
from multiprocessing import Pool
from src.FileIO import save_stellar_data
from src.NumericalIntegration import solve_bvp, rho_index, T_index, M_index, L_index, tau_index
from src.Units import g, cm, million_K
from src.StellarStructureEquations import StellarConfiguration
from src.FileIO import load_stellar_data


R_index, rho_c_index, T_c_index, M_surface_index, L_surface_index, tau_surface_index, T_surface_index = np.arange(7)


def predict_rho_c(T_c, T_c_values, rho_c_values, degree=1):
    """
    Predict the central density based on the given data.

    :param T_c: The central temperature to predict the central density for.
    :param T_c_values: The list of computed central temperature values.
    :param rho_c_values: The corresponding list of computed central density values.
    :param degree: The degree of polynomial to use (in the log-log plot).
    :return: The predicted central density.
    """
    # If there are no data points, then just do a basic prediction
    if len(rho_c_values) == 0:
        return 100 * g / cm ** 3

    # polynomial fit with the last degree + 1 points, or all the points if there are less than degree + 1 points
    x_data = np.log(T_c_values[-(degree + 1):])
    y_data = np.log(rho_c_values[-(degree + 1):])
    polynomial = np.polyfit(x_data, y_data, deg=len(x_data) - 1)
    rho_c_guess = np.exp(np.poly1d(polynomial)(np.log(T_c)))

    # Ensure rho_c_guess is positive
    return rho_c_guess if rho_c_guess > 0 else rho_c_values[-1] / 2


def get_aggregated_data(r_values, state_values):
    """
    Extracts the aggregated summary statistics about this star from its detailed data.

    :param r_values: The radius values.
    :param state_values: The state values.
    :return: The aggregated summary statistics for this star.
    """
    return np.array([r_values[-1], state_values[rho_index, 0], state_values[T_index, 0], state_values[M_index, -1],
                     state_values[L_index, -1], state_values[tau_index, -1], state_values[T_index, -1]])


def calculate_stellar_data(T_c_values, config=StellarConfiguration()):
    """
    Calculates aggregated stellar data for the given central temperature values.
    This function does not use multiprocessing.

    :param T_c_values: The list of central Temperatures to calculate stellar data for.
    :param config: The stellar configuration to use.
    :return:
    """
    stellar_data = None
    for i, T_c in enumerate(T_c_values):
        print(i)
        if i == 0:
            error, r_values, state_values = solve_bvp(T_c)
            stellar_data = get_aggregated_data(r_values, state_values)[:, np.newaxis]
            continue

        rho_c_guess = predict_rho_c(T_c, stellar_data[T_c_index, :], stellar_data[rho_c_index, :])
        error, r_values, state_values = solve_bvp(T_c, rho_c_guess=rho_c_guess, config=config)
        stellar_data = np.column_stack((stellar_data, get_aggregated_data(r_values, state_values)))
        print("Expected:", rho_c_guess, ", Actual:", state_values[rho_index, 0])
    return stellar_data


def generate_stars(T_c_values=np.linspace(4 * million_K, 40 * million_K, 20), config=StellarConfiguration(),
                   file_name=None, threads=4):
    """
    Generates and saves aggregated stellar data for the given list of central temperature values.

    :param T_c_values: The central temperature values of the stars to generate.
    :param config: The StellarConfiguration to use.
    :param file_name: The file name to save the aggregated stellar data.
    :param threads: The number of threads to use.
    :return: The aggregated stellar data.
    """
    if threads > 1:
        params = [(T_c_values[i * len(T_c_values) // threads:(i + 1) * len(T_c_values) // threads], config)
                  for i in range(threads)]
        pool = Pool(threads)
        stellar_data_segments = pool.starmap(calculate_stellar_data, params)
        stellar_data = np.column_stack(tuple(stellar_data_segments))
    else:
        stellar_data = calculate_stellar_data(T_c_values, config)

    save_stellar_data(stellar_data, file_name)
    return stellar_data


def test_rho_c_predictions(file_name='standard_stellar_data', degree=1):
    """
    Compares the predictions that would have been made for the central density to the actual values for the given
    aggregated stellar data.

    :param file_name: The file to load the aggregated stellar data from.
    :param degree: The degree of the polynomial fit to apply.
    :return: The mean percent error.
    """
    stellar_data = load_stellar_data(file_name)
    T_c_vals = stellar_data[T_c_index, :]
    rho_c_vals = stellar_data[rho_c_index, :]
    predictions = []
    for i in range(degree + 1, len(rho_c_vals)):
        predictions.append(predict_rho_c(T_c_vals[i], T_c_vals[:i], rho_c_vals[:i], degree))
    errors = rho_c_vals[degree + 1:] - np.array(predictions)
    percent_errors = errors / rho_c_vals[degree + 1:]
    mean_percent_error = np.mean(np.abs(percent_errors))
    print('Mean Error:', np.mean(np.abs(errors)) / (g / cm ** 3), 'g/cm^3')
    print('Mean Percent Error:', mean_percent_error)
    return mean_percent_error


if __name__ == '__main__':
    test_rho_c_predictions()
