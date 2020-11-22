import numpy as np
from numpy import pi
from scipy import integrate
from src.Units import m, kg, g, cm, M_sun, R_sun
from src.Constants import sigma
from src.Utils import find_zeros_index, interpolate
from src.StellarStructureEquations import StellarConfiguration, rho_index, T_index, M_index, L_index, tau_index

"""
The numerical integration is all done in vector form, where the vector of interest is (rho, T, M, L, tau).
This vector is referred to as the state vector throughout.

The state matrix is the set of all state vectors over all values of the radius (given the variable name state_values).
This will be a 5xN matrix, where N is the number of radius values. 

The indices are referenced using the following variables for readability.
"""


def get_remaining_optical_depth(r, state, kappa_value=None, rho_prime_value=None, config=StellarConfiguration()):
    """
    Calculates an estimate of the remaining optical depth from the current radius to infinity.
    Once this value is sufficiently small, integration can be terminated and the value of optical depth can be assumed
    to be approximately equivalent to the optical depth at infinity.

    :param r: The current radius.
    :param state: The current state vector.
    :param kappa_value: The current optical depth. Can optionally be provided to prevent redoing the calculation.
    :param rho_prime_value: The current derivative of density (with respect to radius).
                            Can optionally be provided to prevent redoing the calculation.
    :param config: The stellar configuration to use.
    """
    rho, T, M, L, _ = state
    if kappa_value is None:
        kappa_value = config.kappa(rho, T)
    if rho_prime_value is None:
        rho_prime_value = config.rho_prime(r, rho, T, M, L)
    return kappa_value * rho ** 2 / np.abs(rho_prime_value)


def truncate_star(r_values, state_values, return_star=False):
    """
    Calculates the (normalized) fractional error between the actual surface luminosity and the surface luminosity
    expected based on the surface radius and temperature.
    The surface radius is interpolated such that the remaining optical depth from the surface is 2/3.

    Can optionally truncate the given stellar data at the surface of the star and add a final data point for the
    surface of the star, where the temperature is manually set to satisfy the boundary condition.

    :param r_values:
    :param state_values:
    :param return_star:
    :return: The fractional surface luminosity error, and optionally the truncated r_values and state_values.
    """
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index, :] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_state = interpolate(state_values, surface_index)

    # calculate the fractional surface luminosity error
    expected_surface_L = 4 * pi * surface_r ** 2 * sigma * surface_state[T_index] ** 4
    error = (surface_state[L_index] - expected_surface_L) / np.sqrt(surface_state[L_index] * expected_surface_L)
    if not return_star:
        return error

    # manually set surface temperature to satisfy boundary condition
    print('Old T', surface_state[T_index])
    surface_state[T_index] = (surface_state[L_index] / (4 * pi * surface_r ** 2 * sigma)) ** (1 / 4)
    print('New T', surface_state[T_index])

    # truncate the star at the surface, and append the manually corrected surface state
    surface_index = int(surface_index)
    r_values = np.append(r_values[:surface_index], surface_r)
    state_values = np.column_stack((state_values[:, :surface_index], surface_state))

    return error, r_values, state_values


def trial_solution(rho_c, T_c, r_0=100 * m, rtol=1e-9, atol=None,
                   return_star=False, optical_depth_threshold=1e-4, mass_threshold=1000 * M_sun,
                   config=StellarConfiguration()):
    """
    Integrates the state of the star from r_0 until the estimated optical depth is below the given threshold.
    The array of radius values and the state matrix are returned along with the fractional surface luminosity error.

    :param rho_c: The central density.
    :param T_c: The central temperature.
    :param r_0: The starting value of the radius. Must be greater than 0 to prevent numerical instabilities.
                Defaults to 100m.
    :param return_star: If True, then the radius values and state matrix will be returned alongside
                        the fractional surface luminosity error.
    :param rtol: The required relative accuracy during integration.
    :param atol: The required absolute accuracy during integration. Defaults to rtol / 1000.
    :param optical_depth_threshold: The value below which the estimated remaining optical depth
                                    must drop for the integration to terminate.
    :param mass_threshold: If the mass of the star increases beyond this value, then integration will be halted.
                           Defaults to 1000 solar masses.
    :param config: The stellar configuration to use.
    :returns: The fractional surface luminosity error, and optionally the array of radius values and the state matrix
    """
    print(rho_c)
    if atol is None:
        atol = rtol / 1000

    # Event based end condition
    def halt_integration(r, state):
        if state[M_index] > mass_threshold:
            return -1
        return get_remaining_optical_depth(r, state, config=config) - optical_depth_threshold

    halt_integration.terminal = True

    # Ending radius is infinity, integration will only be halted via the halt_integration event
    # Not sure what good values for atol and rtol are, but these seem to work well
    result = integrate.solve_ivp(config.get_state_derivative, (r_0, np.inf),
                                 config.get_initial_conditions(rho_c, T_c, r_0=r_0),
                                 events=halt_integration, atol=atol, rtol=rtol)
    # noinspection PyUnresolvedReferences
    r_values, state_values = result.t, result.y

    return truncate_star(r_values, state_values, return_star=return_star)


def solve_bvp(T_c,
              rho_c_guess=100 * g / cm ** 3, confidence=0.9,
              rho_c_min=0.3 * g / cm ** 3, rho_c_max=4e6 * g / cm ** 3,
              rho_c_tol=1e-20 * kg / m ** 3,
              rtol=1e-11, optical_depth_threshold=1e-4,
              config=StellarConfiguration()):
    """
    Solves for the structure of a star with the given central temperature using the point and shoot method.

    This uses the bisection algorithm, with a modification based on confidence. The higher the confidence, the quicker
    convergence will be to rho_c_guess. Once rho_c_guess falls outside the interval of interest,
    simple bisection is used. Too low of a confidence will cause this to reduce to simple bisection, and too high of a
    confidence will likely cause rho_c_guess to fall outside the range of interest too fast leaving an unnecessarily
    large remaining search space.

    :param T_c: The central temperature.
    :param rho_c_guess: A guess for the central density.
    :param confidence: The confidence of the guess. Must be between 0.5 (no confidence) and 1 (perfect confidence).
    :param rho_c_min: The minimum possible central density.
    :param rho_c_max: The maximum possible central density.
    :param rho_c_tol: The tolerance within which the central density must be determined for integration to end.
    :param rtol: The rtol to use.
    :param optical_depth_threshold: The optical_depth_threshold to use.
    :param config: The stellar configuration to use.
    :return: The resulting fractional luminosity error, r_values and state_values of the converged stellar solution.
    """
    args = dict(T_c=T_c, rtol=rtol, optical_depth_threshold=optical_depth_threshold, config=config)
    if confidence < 0.5:
        raise Exception("Confidence must be at least 0.5!")
    if confidence >= 1:
        raise Exception("Confidence must be less than 1!")

    y_guess = trial_solution(rho_c_guess, **args)
    if y_guess == 0:
        return rho_c_guess

    y0 = trial_solution(rho_c_min, **args)
    if y0 == 0:
        return rho_c_min
    if y0 < 0 < y_guess:
        rho_c_low, rho_c_high = rho_c_min, rho_c_guess
        bias_high = True
        bias_low = False
    elif y_guess < 0 < y0:
        rho_c_low, rho_c_high = rho_c_guess, rho_c_min
        bias_low = True
        bias_high = False
    else:
        y1 = trial_solution(rho_c_max, **args)
        if y1 == 0:
            return rho_c_max
        if y1 < 0 < y_guess:
            rho_c_low, rho_c_high = rho_c_max, rho_c_guess
            bias_high = True
            bias_low = False
        elif y_guess < 0 < y1:
            rho_c_low, rho_c_high = rho_c_guess, rho_c_max
            bias_low = True
            bias_high = False
        else:
            print("Retrying with larger rho_c interval for", T_c)
            # set confidence to be much higher since we know that the other boundary will be even further from the guess
            return solve_bvp(T_c, rho_c_guess, confidence=(confidence + 4) / 5,
                             rho_c_min=rho_c_min / 1000, rho_c_max=1000 * rho_c_max,
                             rho_c_tol=rho_c_tol,
                             rtol=rtol,
                             optical_depth_threshold=optical_depth_threshold,
                             config=config)

    while np.abs(rho_c_high - rho_c_low) / 2 > rho_c_tol:
        if bias_low:
            rho_c_guess = confidence * rho_c_low + (1 - confidence) * rho_c_high
            if rho_c_guess == rho_c_low or rho_c_guess == rho_c_high:
                print('Reached limits of numerical precision for rho_c')
                break
            y_guess = trial_solution(rho_c_guess, **args)
            if y_guess == 0:
                return rho_c_guess
            if y_guess < 0:
                rho_c_low = rho_c_guess
                bias_low = False  # ignore initial guess bias now that it is no longer the low endpoint
            elif y_guess > 0:
                rho_c_high = rho_c_guess
        elif bias_high:
            rho_c_guess = (1 - confidence) * rho_c_low + confidence * rho_c_high
            if rho_c_guess == rho_c_low or rho_c_guess == rho_c_high:
                print('Reached limits of numerical precision for rho_c')
                break
            y_guess = trial_solution(rho_c_guess, **args)
            if y_guess == 0:
                return rho_c_guess
            if y_guess < 0:
                rho_c_low = rho_c_guess
            elif y_guess > 0:
                rho_c_high = rho_c_guess
                bias_high = False  # ignore initial guess bias now that it is no longer the high endpoint
        else:
            rho_c_guess = (rho_c_low + rho_c_high) / 2
            if rho_c_guess == rho_c_low or rho_c_guess == rho_c_high:
                print('Reached limits of numerical precision for rho_c')
                break
            y_guess = trial_solution(rho_c_guess, **args)
            if y_guess == 0:
                return rho_c_guess
            if y_guess < 0:
                rho_c_low = rho_c_guess
            elif y_guess > 0:
                rho_c_high = rho_c_guess

    rho_c = (rho_c_high + rho_c_low) / 2

    # if solution failed to converge, recurse with greater accuracy
    if np.abs(y_guess) > 1000:
        print('Retrying with higher accuracy for', T_c)
        return solve_bvp(T_c, rho_c, confidence=0.99,  # confidence is extremely high now
                         rho_c_min=rho_c_min, rho_c_max=rho_c_max,
                         rho_c_tol=rho_c_tol,
                         rtol=rtol * 100,
                         optical_depth_threshold=optical_depth_threshold)

    # Generate and return final star
    return trial_solution(rho_c, T_c, return_star=True, rtol=rtol,
                          optical_depth_threshold=optical_depth_threshold)
