import numpy as np
from src.Constants import *
from src.Utils import find_zeros, interpolate
from src.StellarStructureEquations import rho_prime, T_prime, M_prime, L_prime, tau_prime, kappa, epsilon

rho_index = 0
T_index = 1
M_index = 2
L_index = 3
tau_index = 4


def get_initial_conditions(rho_c, T_c, r_0=1):
    """
    Calculates the initial vector of variables.
    :param rho_c: The central density.
    :param T_c: The central temperature.
    :param r_0: The starting radius. r_0 = 0 cannot be used due to numerical instabilities. Defaults to r_0 = 1m.
    :return:
    """
    M_c = (4 / 3) * pi * r_0 ** 3 * rho_c
    L_c = M_c * epsilon(rho_c, T_c)
    kappa_c = kappa(rho_c, T_c)
    tau_c = kappa_c * rho_c * r_0  # TODO: is initial condition for optical depth correct
    return np.array([rho_c, T_c, M_c, L_c, tau_c])


def get_state_derivative(r, state, return_kappa=False):
    """
    Calculates the derivatives of all 5 variables and returns the result, in vector form.
    If return_kappa is set to True, then kappa will also be returned as the second item of a tuple
    """
    rho, T, M, L, tau = state
    kappa_value = kappa(rho, T)

    T_prime_value = T_prime(r, rho, T, M, L, kappa_value=kappa_value)
    rho_prime_value = rho_prime(r, rho, T, M, L, T_prime_value=T_prime_value)
    M_prime_value = M_prime(r, rho)
    L_prime_value = L_prime(r, rho, T, M_prime_value=M_prime_value)
    tau_prime_value = tau_prime(rho, T, kappa_value=kappa_value)

    state_derivative = np.array([rho_prime_value, T_prime_value, M_prime_value, L_prime_value, tau_prime_value])
    return state_derivative, kappa_value if return_kappa else state_derivative


def get_remaining_optical_depth(r, rho, T, M, L, kappa_value=None, rho_prime_value=None):
    """
    Calculates an estimate of the remaining optical depth from the current radius to infinity.
    Once this value is sufficiently small, integration can be terminated and the value of optical depth can be assumed
    to be approximately equivalent to the optical depth at infinity.
    """
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    if rho_prime_value is None:
        rho_prime_value = rho_prime(r, rho, T, M, L)
    return kappa_value * rho ** 2 / np.abs(rho_prime_value)


def surface_L_error(r_values, state_values):
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros(tau_infinity - state_values[tau_index] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_T = interpolate(state_values[T_index, :], surface_index)
    surface_L = interpolate(state_values[L_index, :], surface_index)

    expected_surface_L = 4 * pi * surface_r ** 2 * sigma * surface_T ** 4
    return (surface_L - expected_surface_L) / np.sqrt(surface_L * expected_surface_L)


def trial_solution(rho_c, T_c, delta_r, r_0=None, optical_depth_threshold=1e-4):
    """
    In the returned matrix of state values, the first index corresponds to the type of variable, and the second index
    corresponds to the radius.
    :param rho_c:
    :param T_c:
    :param delta_r:
    :param r_0:
    :param optical_depth_threshold:
    :return:
    """
    if r_0 is None:
        r_0 = delta_r / 10 ** 3

    r = r_0
    state = get_initial_conditions(rho_c, T_c, r_0=r_0)

    # Note even though the values for r=0 are not used as the first step,
    # they can still be included in the list of values
    r_values = [0, r_0]
    state_values = [np.array([rho_c, T_c, 0, 0, 0]), state]  # TODO: is initial condition for optical depth correct?

    while state[M_index] < 10 ** 3 * M_sun:
        # TODO: upgrade to Runge-Kutta
        state_derivative, kappa_value = get_state_derivative(r, state, return_kappa=True)

        r += delta_r
        state += state_derivative * delta_r

        r_values += [r]
        state_values += [state]

        if get_remaining_optical_depth(r, *state[:-1], kappa_value=kappa_value,
                                       rho_prime_value=state_derivative[rho_index]) < optical_depth_threshold:
            break

    r_values = np.array(r_values)
    state_values = np.column_stack(state_values)

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

    return r_values, state_values
