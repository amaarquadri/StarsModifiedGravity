import numpy as np
from numpy import pi
from scipy import integrate
from src.Units import M_sun, R_sun
from src.Constants import sigma
from src.Utils import find_zeros_index, interpolate, print_state
from src.StellarStructureEquations import rho_prime, T_prime, M_prime, L_prime, tau_prime, kappa, epsilon

"""
The numerical integration is all done in vector form, where the vector of interest is (rho, T, M, L, tau).
This vector is referred to as the state vector throughout.

The state matrix is the set of all state vectors over all values of the radius (given the variable name state_values).
This will be a 5xN matrix, where N is the number of radius values. 

The indices are referenced using the following variables for readability.
"""
rho_index, T_index, M_index, L_index, tau_index = 0, 1, 2, 3, 4


def get_initial_conditions(rho_c, T_c, r_0=1):
    """
    Calculates the initial state vector.

    :param rho_c: The central density.
    :param T_c: The central temperature.
    :param r_0: The starting radius. r_0 = 0 cannot be used due to numerical instabilities. Defaults to r_0 = 1m.
    :return: The state vector at r = r_0 for the given central density and temperature.
    """
    M_c = (4 / 3) * pi * r_0 ** 3 * rho_c
    L_c = M_c * epsilon(rho_c, T_c)
    kappa_c = kappa(rho_c, T_c)
    tau_c = kappa_c * rho_c * r_0  # TODO: is initial condition for optical depth correct
    return np.array([rho_c, T_c, M_c, L_c, tau_c])


def get_state_derivative(r, state, return_kappa=False):
    """
    Calculates the elementwise derivative of the state vector.

    :param r: The current radius.
    :param state: The state vector at the given radius.
    :param return_kappa: If set to True, then the opacity will be returned as the second item of a tuple.
    :return: The elementwise derivative of the state vector, and optionally the optical depth as well.
    """
    rho, T, M, L, _ = state
    kappa_value = kappa(rho, T)

    T_prime_value = T_prime(r, rho, T, M, L, kappa_value=kappa_value)
    rho_prime_value = rho_prime(r, rho, T, M, L, T_prime_value=T_prime_value)
    M_prime_value = M_prime(r, rho)
    L_prime_value = L_prime(r, rho, T, M_prime_value=M_prime_value)
    tau_prime_value = tau_prime(rho, T, kappa_value=kappa_value)

    state_derivative = np.array([rho_prime_value, T_prime_value, M_prime_value, L_prime_value, tau_prime_value])
    return (state_derivative, kappa_value) if return_kappa else state_derivative


def get_state_derivative_rk4(r, state, delta_r, return_kappa=False):
    """
    Estimates the elementwise derivative of the state vector using the 4th order Runge-Kutta method.
    This will provide a more accurate estimate of the average derivative of the state vector between r and r + delta_r.

    :param r: The current radius.
    :param state: The state vector at the given radius.
    :param delta_r: The size of the step in the radius.
    :param return_kappa: If set to True, then the opacity will be returned as the second item of a tuple.
    :return: The elementwise derivative of the state vector, and optionally the optical depth as well.
    """
    state_prime_0, kappa_0 = get_state_derivative(r, state, return_kappa=True)
    state_prime_1, kappa_1 = get_state_derivative(r + delta_r / 2, state + state_prime_0 * delta_r / 2,
                                                  return_kappa=True)
    state_prime_2, kappa_2 = get_state_derivative(r + delta_r / 2, state + state_prime_1 * delta_r / 2,
                                                  return_kappa=True)
    state_prime_3, kappa_3 = get_state_derivative(r + delta_r, state + state_prime_2 * delta_r, return_kappa=True)

    state_prime_rk4 = (state_prime_0 + 2 * state_prime_1 + 2 * state_prime_2 + state_prime_3) / 6
    if not return_kappa:
        return state_prime_rk4

    kappa_rk4 = (kappa_0 + 2 * kappa_1 + 2 * kappa_2 + kappa_3) / 6
    return state_prime_rk4, kappa_rk4


def get_remaining_optical_depth(r, state, kappa_value=None, rho_prime_value=None):
    """
    Calculates an estimate of the remaining optical depth from the current radius to infinity.
    Once this value is sufficiently small, integration can be terminated and the value of optical depth can be assumed
    to be approximately equivalent to the optical depth at infinity.

    :param r: The current radius.
    :param state: The current state vector.
    :param kappa_value: The current optical depth. Can optionally be provided to prevent redoing the calculation.
    :param rho_prime_value: The current derivative of density (with respect to radius).
                            Can optionally be provided to prevent redoing the calculation.
    """
    rho, T, M, L, _ = state
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    if rho_prime_value is None:
        rho_prime_value = rho_prime(r, rho, T, M, L)
    return kappa_value * rho ** 2 / np.abs(rho_prime_value)


def surface_L_error(r_values, state_values):
    """
    Calculates the (normalized) fractional error between the actual surface luminosity and the surface luminosity
    expected based on the surface radius and temperature.
    The surface radius is interpolated such that the remaining optical depth from the surface is 2/3.

    :param r_values: The vector of r values.
    :param state_values: The matrix of state values.
    :return: The fractional error between the actual and expected surface luminosity.
    """
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_state = interpolate(state_values, surface_index)

    expected_surface_L = 4 * pi * surface_r ** 2 * sigma * surface_state[T_index] ** 4
    return (surface_state[L_index] - expected_surface_L) / np.sqrt(surface_state[L_index] * expected_surface_L)


def trial_solution(rho_c, T_c, delta_r, r_0=None, optical_depth_threshold=1e-4):
    """
    Integrates the state of the star from r_0 until the estimated optical depth is below the given threshold.
    The array of radius values and the state matrix are returned along with the fractional surface luminosity error.

    :param rho_c: The central density.
    :param T_c: The central temperature.
    :param delta_r: The size of the radius steps to take during integration.
    :param r_0: The starting value of the radius. Must be greater than 0 to prevent numerical instabilities.
                Defaults to delta_r / 1000.
    :param optical_depth_threshold: The value below which the estimated remaining optical depth
                                    must drop for the integration to terminate.
    :returns: The array of radius values, the state matrix, the fractional surface luminosity error
    """
    if r_0 is None:
        r_0 = delta_r / 10 ** 3

    r = r_0
    state = get_initial_conditions(rho_c, T_c, r_0=r_0)

    # Note even though the values for r=0 are not used as the first step,
    # they can still be included in the list of values
    r_values = np.array([0, r_0])
    # TODO: is initial condition for optical depth correct?
    state_values = np.column_stack((np.array([rho_c, T_c, 0, 0, 0]), state))

    count = 0
    # only save the data a fixed number of times per solar radius
    save_frequency = np.ceil(R_sun / delta_r / 1000)
    while state[M_index] < 10 ** 3 * M_sun:
        state_derivative, kappa_value = get_state_derivative_rk4(r, state, delta_r, return_kappa=True)

        r += delta_r
        state += state_derivative * delta_r

        if count % save_frequency == 0:
            r_values = np.append(r_values, r)
            state_values = np.column_stack((state_values, state))

            # only bother to check the exit condition a fixed number of times per unit solar radius
            depth = get_remaining_optical_depth(r, state, kappa_value=kappa_value,
                                                rho_prime_value=state_derivative[rho_index])

            if depth < optical_depth_threshold:
                break

        count += 1

    error = surface_L_error(r_values, state_values)
    return r_values, state_values, error


# noinspection PyUnresolvedReferences
def trial_solution_rk45(rho_c, T_c, delta_r, r_0=None, optical_depth_threshold=1e-4):
    if r_0 is None:
        r_0 = delta_r / 10 ** 3

    # Note even though the values for r=0 are not used as the first step,
    # they can still be included in the list of values
    r_values = np.array([0, r_0])
    # TODO: is initial condition for optical depth correct?
    state_values = np.column_stack((np.array([rho_c, T_c, 0, 0, 0]), get_initial_conditions(rho_c, T_c, r_0=r_0)))

    while state_values[M_index, -1] < 10 ** 3 * M_sun:
        result = integrate.solve_ivp(get_state_derivative, (r_values[-1], r_values[-1] + 0.5 * R_sun),
                                     state_values[:, -1], max_step=delta_r)
        r_values = np.concatenate((r_values, result.t[1:]))
        state_values = np.column_stack((state_values, result.y[:, 1:]))

        depth = get_remaining_optical_depth(r_values[-1], state_values[:, -1])
        print_state(r_values[-1], state_values[:, -1], depth)

        # if remaining optical depth is too low, then division by 0 results in Nan
        if np.isnan(depth) or depth < optical_depth_threshold:
            break

    error = surface_L_error(r_values, state_values)
    return r_values, state_values, error


def solve_numerically(T_c, delta_r, error_threshold=1e-4):
    low_rho = 0.3
    high_rho = 500

    low_error = trial_solution(low_rho, T_c, delta_r)[-1]

    r_values = None
    state_values = None

    while high_rho - low_rho > error_threshold:
        # TODO: run 4 computations in parallel (i.e. cut search space by a factor of 5 each time instead of 2)
        rho_guess = (low_rho + high_rho) / 2
        r_values, state_values, error = trial_solution(rho_guess, T_c, delta_r)

        if np.sign(error) == np.sign(low_error):
            low_rho = rho_guess
            low_error = error
        else:
            high_rho = rho_guess

    # TODO: do we need to manually apply the boundary condition?
    return r_values, state_values
