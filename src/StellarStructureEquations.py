import numpy as np
from numpy import pi
from src.Constants import G, a, c, gamma, m_p, m_e, h_bar, k_b, kappa_es_coefficient, kappa_ff_coefficient, \
    kappa_H_minus_coefficient, epsilon_proton_proton_coefficient, epsilon_CNO_coefficient


X_default = 0.7
Z_default = 0.031  # 0.034070001061466126  # 0.014
Y_default = 1 - Z_default - X_default  # 0.274


def rho_prime(r, rho, T, M, L, T_prime_value=None):
    """
    The T_prime_value can optionally be provided to prevent redoing the calculation.
    """
    if T_prime_value is None:
        T_prime_value = T_prime(r, rho, T, M, L)
    return -(G * M * rho / r ** 2 + dP_by_dT(rho, T) * T_prime_value) / dP_by_drho(rho, T)


def T_prime(r, rho, T, M, L, kappa_value=None):
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    radiative = 3 * kappa_value * rho * L / (16 * pi * a * c * T ** 3 * r ** 2)
    convective = (1 - 1 / gamma) * T * G * M * rho / (P(rho, T) * r ** 2)
    return -np.minimum(radiative, convective)


def T_prime_radiative(r, rho, T, L, kappa_value=None):
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    return -3 * kappa_value * rho * L / (16 * pi * a * c * T ** 3 * r ** 2)


def T_prime_convective(r, rho, T, M):
    return -(1 - 1 / gamma) * T * G * M * rho / (P(rho, T) * r ** 2)


def is_convective(r, rho, T, M, L, kappa_value=None):
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    return T_prime_convective(r, rho, T, M) > T_prime_radiative(r, rho, T, L, kappa_value=kappa_value)


def M_prime(r, rho):
    return 4 * pi * r ** 2 * rho


def L_prime(r, rho, T, M_prime_value=None):
    if M_prime_value is None:
        M_prime_value = M_prime(r, rho)
    return M_prime_value * epsilon(rho, T)


def L_proton_proton_prime(r, rho, T, M_prime_value=None):
    if M_prime_value is None:
        M_prime_value = M_prime(r, rho)
    return M_prime_value * epsilon_proton_proton(rho, T)


def L_CNO_prime(r, rho, T, M_prime_value=None):
    if M_prime_value is None:
        M_prime_value = M_prime(r, rho)
    return M_prime_value * epsilon_CNO(rho, T)


def tau_prime(rho, T, kappa_value=None):
    if kappa_value is None:
        kappa_value = kappa(rho, T)
    return kappa_value * rho


def P(rho, T):
    return P_degeneracy(rho) + P_gas(rho, T) + P_photon(T)


def P_degeneracy(rho):
    return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (5 * m_e) * (rho / m_p) ** (5 / 3)


def P_gas(rho, T):
    return rho * k_b * T / (mu() * m_p)


def P_photon(T):
    return (1 / 3) * a * T ** 4


def dP_by_drho(rho, T):
    return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (3 * m_e * m_p) * (rho / m_p) ** (2 / 3) + k_b * T / (mu() * m_p)


def dP_by_dT(rho, T):
    return rho * k_b / (mu() * m_p) + (4 / 3) * a * T ** 3


def kappa(rho, T):
    return (1 / kappa_H_minus(rho, T) + 1 / np.maximum(kappa_es(), kappa_ff(rho, T))) ** -1


def kappa_es(X=X_default):
    return kappa_es_coefficient * (X + 1)


def kappa_ff(rho, T, Z=Z_default):
    # TODO: shouldn't T be divided by some power of 10?
    return kappa_ff_coefficient * (Z + 0.0001) * (rho / 10 ** 3) ** 0.7 * T ** -3.5


def kappa_H_minus(rho, T, Z=Z_default):
    # TODO: shouldn't T be divided by some power of 10?
    return kappa_H_minus_coefficient * (Z / 0.02) * (rho / 10 ** 3) ** 0.5 * T ** 9


def epsilon(rho, T):
    return epsilon_proton_proton(rho, T) + epsilon_CNO(rho, T)


def epsilon_proton_proton(rho, T, X=X_default):
    return epsilon_proton_proton_coefficient * X ** 2 * (rho / 10 ** 5) * (T / 10 ** 6) ** 4


def epsilon_CNO(rho, T, X=X_default, X_CNO=None):
    if X_CNO is None:
        X_CNO = 0.03 * X
    return epsilon_CNO_coefficient * X * X_CNO * (rho / 10 ** 5) * (T / 10 ** 6) ** 19.9


def mu(X=X_default, Y=Y_default, Z=Z_default):
    return (2 * X + 0.75 * Y + 0.5 * Z) ** -1
