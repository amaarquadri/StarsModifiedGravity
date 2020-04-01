import numpy as np
from numpy import pi
from scipy import integrate
from src.Units import m, kg, g, cm, M_sun, R_sun
from src.Constants import sigma
from src.Utils import find_zeros_index, interpolate
from src.StellarStructureEquations import rho_prime, T_prime, M_prime, L_prime, tau_prime, kappa, epsilon

"""
The numerical integration is all done in vector form, where the vector of interest is (rho, T, M, L, tau).
This vector is referred to as the state vector throughout.

The state matrix is the set of all state vectors over all values of the radius (given the variable name state_values).
This will be a 5xN matrix, where N is the number of radius values. 

The indices are referenced using the following variables for readability.
"""
rho_index, T_index, M_index, L_index, tau_index = np.arange(5)


def get_initial_conditions(rho_c, T_c, r_0=1 * m):
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


def truncate_star(r_values, state_values):
    """
    Truncates the given stellar data at the surface of the star. Additionally, this adds a final data point for the
    surface of the star, where the temperature is manually set to satisfy the boundary condition.

    :param r_values:
    :param state_values:
    :return: The truncated r_values and state_values.
    """
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index, :] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_state = interpolate(state_values, surface_index)

    # manually set surface temperature to satisfy boundary condition
    surface_state[M_index] = state_values[M_index, -1]
    surface_state[L_index] = state_values[L_index, -1]
    surface_state[T_index] = (surface_state[L_index] / (4 * pi * surface_r ** 2 * sigma)) ** (1 / 4)

    surface_index = int(surface_index)
    r_values = np.append(r_values[:surface_index], surface_r)
    state_values = np.column_stack((state_values[:, :surface_index], surface_state))

    return r_values, state_values


# noinspection PyUnresolvedReferences
def trial_solution(rho_c, T_c, r_0=100 * m, rtol=1e-9, atol=None,
                   return_star=False, optical_depth_threshold=1e-4, mass_threshold=1000 * M_sun):
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
    :returns: The fractional surface luminosity error, and optionally the array of radius values and the state matrix
    """
    if atol is None:
        atol = rtol / 1000

    # Event based end condition
    def halt_integration(r, state):
        if state[M_index] > mass_threshold:
            return -1
        return get_remaining_optical_depth(r, state) - optical_depth_threshold

    halt_integration.terminal = True

    # Ending radius is infinity, integration will only be halted via the halt_integration event
    # Not sure what good values for atol and rtol are, but these seem to work well
    result = integrate.solve_ivp(get_state_derivative, (r_0, np.inf), get_initial_conditions(rho_c, T_c, r_0=r_0),
                                 events=halt_integration, atol=atol, rtol=rtol)
    r_values = result.t
    state_values = result.y

    error = surface_L_error(r_values, state_values)
    return (r_values, state_values, error) if return_star else error


def solve_bvp(T_c,
              rho_c_guess=100 * g / cm ** 3, confidence=0.9,
              rho_c_min=0.3 * g / cm ** 3, rho_c_max=4e9 * g / cm ** 3,
              high_accuracy_threshold=10 * kg / m ** 3, rho_c_tol=1e-7 * kg / m ** 3,
              max_rtol=1e-7, min_rtol=1e-10,
              max_optical_depth_threshold=1e-3, min_optical_depth_threshold=1e-4):
    """
    Solves for the structure of a star with the given central temperature using the point and shoot method.

    This uses the bisection algorithm, with a modification based on confidence. The higher the confidence, the quicker
    convergence will be to rho_c_guess. Once rho_c_guess falls outside the interval of interest,
    simple bisection is used. Too low of a confidence will cause this to reduce to simple bisection, and too high of a
    confidence will likely cause rho_c_guess to fall outside the range of interest too fast leaving an unnecessarily
    large remaining search space.

    This algorithm adaptively increases the trial solution integration accuracy. The low accuracy one is used
    initially until high_accuracy_threshold is reached. Then integration accuracy is increase logarithmically
    proportionally as the range of considered central density values converges to within rho_c_tol. Both the rtol and
    optical_depth_thresholds are improved from their max to their min provided values.

    :param T_c: The central temperature.
    :param rho_c_guess: A guess for the central density.
    :param confidence: The confidence of the guess. Must be between 0.5 (no confidence) and 1 (perfect confidence).
    :param rho_c_min: The minimum possible central density.
    :param rho_c_max: The maximum possible central density.
    :param high_accuracy_threshold: The value below which the range of rho_c values must drop below before switching to
    :param rho_c_tol: The tolerance within which the central density must be determined for integration to end.
                                    high accuracy mode. Defaults to 100 * rho_c_tol.
    :param max_rtol: The starting rtol to use.
    :param min_rtol: The rtol will gradually improve up to this value as the solution converges.
    :param max_optical_depth_threshold: The starting optical_depth_threshold to use.
    :param min_optical_depth_threshold: The optical_depth_threshold will gradually improve up to this value
                                        as the solution converges.
    :return: The resulting r_values and state_values of the converged stellar solution.
    """
    if confidence < 0.5:
        raise Exception("Confidence must be at least 0.5!")
    if confidence >= 1:
        raise Exception("Confidence must be less than 1!")

    # The rtol and optical_depth_threshold values used will be logarithmically interpolated between
    # the min and max as the solution converges
    def get_args(rho_c_range):
        rtol = 10 ** np.interp(np.log10(rho_c_range),
                               np.log10([rho_c_tol, high_accuracy_threshold]),
                               np.log10([min_rtol, max_rtol]))
        optical_depth_threshold = 10 ** np.interp(np.log10(rho_c_range),
                                                  np.log10([rho_c_tol, 1]),
                                                  np.log10([min_optical_depth_threshold, max_optical_depth_threshold]))
        return dict(rtol=rtol, optical_depth_threshold=optical_depth_threshold)

    y_guess = trial_solution(rho_c_guess, T_c, rtol=min_rtol, optical_depth_threshold=min_optical_depth_threshold)
    if y_guess == 0:
        return rho_c_guess

    y0 = trial_solution(rho_c_min, T_c, rtol=max_rtol, optical_depth_threshold=max_optical_depth_threshold)
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
        y1 = trial_solution(rho_c_max, T_c, rtol=max_rtol, optical_depth_threshold=max_optical_depth_threshold)
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
            return solve_bvp(T_c, rho_c_guess, confidence=0.99,
                             rho_c_min=rho_c_min / 1000, rho_c_max=1000 * rho_c_max,
                             high_accuracy_threshold=high_accuracy_threshold, rho_c_tol=rho_c_tol,
                             max_rtol=max_rtol, min_rtol=min_rtol,
                             max_optical_depth_threshold=max_optical_depth_threshold,
                             min_optical_depth_threshold=min_optical_depth_threshold)

    while np.abs(rho_c_high - rho_c_low) / 2 > rho_c_tol:
        # Calculate the rtol and optical_depth_threshold values to use for this iteration
        args = get_args(np.abs(rho_c_high - rho_c_low) / 2)

        if bias_low:
            rho_c_guess = confidence * rho_c_low + (1 - confidence) * rho_c_high
            y_guess = trial_solution(rho_c_guess, T_c, **args)
            if y_guess == 0:
                return rho_c_guess
            if y_guess < 0:
                rho_c_low = rho_c_guess
                bias_low = False  # ignore initial guess bias now that it is no longer the low endpoint
            elif y_guess > 0:
                rho_c_high = rho_c_guess
        elif bias_high:
            rho_c_guess = (1 - confidence) * rho_c_low + confidence * rho_c_high
            y_guess = trial_solution(rho_c_guess, T_c, **args)
            if y_guess == 0:
                return rho_c_guess
            if y_guess < 0:
                rho_c_low = rho_c_guess
            elif y_guess > 0:
                rho_c_high = rho_c_guess
                bias_high = False  # ignore initial guess bias now that it is no longer the high endpoint
        else:
            rho_c_guess = (rho_c_low + rho_c_high) / 2
            y_guess = trial_solution(rho_c_guess, T_c, **args)
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
        return solve_bvp(T_c, rho_c, confidence=0.99, rho_c_min=rho_c_min, rho_c_max=rho_c_max,
                         high_accuracy_threshold=high_accuracy_threshold, rho_c_tol=rho_c_tol,
                         max_rtol=max_rtol * 100, min_rtol=min_rtol * 100,
                         max_optical_depth_threshold=max_optical_depth_threshold,
                         min_optical_depth_threshold=min_optical_depth_threshold)

    # Generate and return final star
    r_values, state_values, error = trial_solution(rho_c, T_c, return_star=True, rtol=min_rtol,
                                                   optical_depth_threshold=min_optical_depth_threshold)
    r_values, state_values = truncate_star(r_values, state_values)
    return r_values, state_values


# OLD CODE


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


def trial_solution_manual(rho_c, T_c, delta_r, r_0=None, optical_depth_threshold=1e-4):
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

            if depth < optical_depth_threshold or r > 0.863869661 * R_sun:  # hardcoded end condition for debugging
                break

        count += 1

    error = surface_L_error(r_values, state_values)
    return r_values, state_values, error
