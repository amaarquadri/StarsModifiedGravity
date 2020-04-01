import numpy as np
from multiprocessing import Pool
from src.FileIO import save_stellar_data
from src.NumericalIntegration import solve_bvp, rho_index, T_index, M_index, L_index, tau_index
from src.Units import g, cm, million_K

R_index, rho_c_index, T_c_index, M_surface_index, L_surface_index, tau_surface_index, T_surface_index = np.arange(7)


def predict_rho_c(T_c, T_c_values, rho_c_values):
    """
    Predict the central density based on the given data.

    :param T_c: The central temperature to predict the central density for.
    :param T_c_values: The list of computed central temperature values.
    :param rho_c_values: The corresponding list of computed central density values.
    :return: The predicted central density.
    """
    # If there are less than 2 data points, then just do a basic prediction
    if len(T_c_values) == 0:
        return 100 * g / cm ** 3
    elif len(rho_c_values) == 1:
        return rho_c_values[0]
    # Normalize by T_c_values[-1] and rho_c_values[-1]
    # m = np.sum((T_c_values - T_c_values[-1]) * (rho_c_values - rho_c_values[-1])) / \
    #     np.sum((T_c_values - T_c_values[-1]) ** 2)
    # rho_c_guess = m * (T_c - T_c_values[-1]) + rho_c_values[-1]

    # Normalize by subtracting T_c_values[-1] and rho_c_values[-1], then taking the log
    # Then apply least squares fit of y=mx, where m = sum(x_i * y_i) / sum(x_i ** 2)
    # p = np.sum(np.log(T_c_values - T_c_values[-1]) * np.log(rho_c_values - rho_c_values[-1])) / \
    #     np.sum(np.log(T_c_values - T_c_values[-1]) ** 2)
    # # np.log(rho_c_guess - rho_c_values[-1]) = p * np.log(T_c - T_c_values[-1])
    # rho_c_guess = (T_c - T_c_values[-1]) ** p + rho_c_values[-1]

    # Linear prediction in log-log graph based on last 2 points only
    # This is more robust to long term changes in the slope of the log-log graph, since it only applies locally
    m = (np.log(rho_c_values[-1]) - np.log(rho_c_values[-2])) / (np.log(T_c_values[-1]) - np.log(T_c_values[-2]))
    rho_c_guess = np.exp(m * (np.log(T_c) - np.log(T_c_values[-1])) + np.log(rho_c_values[-1]))

    # Ensure rho_c_guess is positive
    return rho_c_guess if rho_c_guess > 0 else rho_c_values[-1] / 2


def get_aggregated_data(r_values, state_values):
    return np.array([r_values[-1], state_values[rho_index, 0], state_values[T_index, 0], state_values[M_index, -1],
                     state_values[L_index, -1], state_values[tau_index, -1], state_values[T_index, -1]])


def calculate_stellar_data(T_c_values):
    """
    Calculates aggregated stellar data using

    :param T_c_values: The list of central Temperatures to calculate stellar data for.
    :return:
    """
    stellar_data = None
    for i, T_c in enumerate(T_c_values):
        print(i)
        if i == 0:
            r_values, state_values = solve_bvp(T_c)
            stellar_data = get_aggregated_data(r_values, state_values)[:, np.newaxis]
            continue

        rho_c_guess = predict_rho_c(T_c, stellar_data[T_c_index, :], stellar_data[rho_c_index, :])
        r_values, state_values = solve_bvp(T_c, rho_c_guess=rho_c_guess)
        stellar_data = np.column_stack((stellar_data, get_aggregated_data(r_values, state_values)))
        print("Expected:", rho_c_guess, ", Actual:", state_values[rho_index, 0])
    return stellar_data


def generate_stars(T_c_values=np.linspace(0.01 * million_K, 36 * million_K, 150), file_name=None, threads=4):
    if threads > 1:
        T_c_values_list = [T_c_values[i * len(T_c_values) // threads:(i + 1) * len(T_c_values) // threads]
                           for i in range(threads)]
        pool = Pool(threads)
        stellar_data_segments = pool.map(calculate_stellar_data, T_c_values_list)
        stellar_data = np.column_stack(tuple(stellar_data_segments))
    else:
        stellar_data = calculate_stellar_data(T_c_values)

    save_stellar_data(stellar_data, file_name)
    return stellar_data
