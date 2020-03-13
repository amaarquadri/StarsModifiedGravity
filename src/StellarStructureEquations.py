from src.Constants import *
import numpy as np


def rho_prime(r, rho, T, M, L, T_prime_value=None):
    """
    The T_prime_value can optionally be provided to prevent redoing the calculation.
    """
    if T_prime_value is None:
        T_prime_value = T_prime(r, rho, T, M, L)
    return -(G * M * rho / (r ** 2) + dP_by_dT(rho, T) * T_prime_value) / dP_by_drho(rho, T)


def T_prime(r, rho, T, M, L, kappa_value=None):
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    return -np.min([3 * kappa_value * rho * L / (16 * pi * a * c * T ** 3 * r ** 2),
                    (1 - 1 / gamma) * T * G * M * rho / (P(rho, T) * r ** 2)])


def M_prime(r, rho):
    return 4 * pi * r ** 2 * rho


def L_prime(r, rho, T, M_prime_value=None):
    if M_prime_value is None:
        M_prime_value = M_prime(r, rho)
    return M_prime_value * epsilon(rho, T)


def tau_prime(rho, T, kappa_value=None):
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    return kappa_value * rho


def P(rho, T):
    return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (5 * m_e) * (rho / m_p) ** (5 / 3) + \
           rho * k_b * T / (mu() * m_p) + (1 / 3) * a * T ** 4


def dP_by_dT(rho, T):
    return rho * k_b / (mu() * m_p) + (4 / 3) * a * T ** 3


def dP_by_drho(rho, T):
    return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (3 * m_e * m_p) * (rho / m_p) ** (2 / 3) + k_b * T / (mu() * m_p)


def kappa(rho, T):
    return (1 / kappa_H_minus(rho, T) + 1 / np.max([kappa_es(), kappa_ff(rho, T)])) ** -1


def kappa_es(X=1):
    return kappa_es_coefficient * (X + 1)


def kappa_ff(rho, T, Z=0.01):
    # TODO: shouldn't T be divided by some power of 10?
    return kappa_ff_coefficient * (Z + 0.0001) * (rho / 10 ** 3) ** 0.7 * T ** -3.5


def kappa_H_minus(rho, T, Z=0.01):
    # TODO: shouldn't T be divided by some power of 10?
    return kappa_H_minus_coefficient * (Z / 0.02) * (rho / 10 ** 3) ** 0.5 * T ** 9


def epsilon(rho, T):
    return epsilon_proton_proton(rho, T) + epsilon_CNO(rho, T)


def epsilon_proton_proton(rho, T, X=1):
    return epsilon_proton_proton_coefficient * X ** 2 * (rho / 10 ** 5) * (T / 10 ** 6) ** 4


def epsilon_CNO(rho, T, X=1, X_CNO=None):
    if X_CNO is None:
        X_CNO = 0.03 * X
    return epsilon_CNO_coefficient * X * X_CNO * (rho / 10 ** 5) * (T / 10 ** 6) ** 19.9


def mu(X=1, Y=0.25, Z=0.01):
    return (2 * X + 0.75 * Y + 0.5 * Z) ** -1