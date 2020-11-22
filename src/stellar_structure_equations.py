import numpy as np
from numpy import pi
from src.constants import G, a, c, gamma, m_p, m_e, h_bar, k_b, kappa_es_coefficient, kappa_ff_coefficient, \
    kappa_H_minus_coefficient, epsilon_proton_proton_coefficient, epsilon_CNO_coefficient
from src.units import m, R_sun

rho_index, T_index, M_index, L_index, tau_index = np.arange(5)
X_default = 0.7
Z_default = 0.031
Y_default = 1 - Z_default - X_default
lambda_small_default = 0
lambda_large_default = np.inf


class StellarConfiguration:
    def __init__(self, X=X_default, Y=Y_default, Z=Z_default,
                 lambda_small=lambda_small_default, lambda_large=lambda_large_default):
        if X + Y + Z != 1:
            raise Exception('X + Y + Z != 1')
        self.X = X
        self.Y = Y
        self.Z = Z
        self.lambda_small = lambda_small
        self.lambda_large = lambda_large

    def get_initial_conditions(self, rho_c, T_c, r_0=1 * m):
        """
        Calculates the initial state vector.

        :param rho_c: The central density.
        :param T_c: The central temperature.
        :param r_0: The starting radius. r_0 = 0 cannot be used due to numerical instabilities. Defaults to r_0 = 1m.
        :return: The state vector at r = r_0 for the given central density and temperature.
        """
        M_c = (4 / 3) * pi * r_0 ** 3 * rho_c
        L_c = M_c * self.epsilon(rho_c, T_c)
        kappa_c = self.kappa(rho_c, T_c)
        tau_c = kappa_c * rho_c * r_0
        return np.array([rho_c, T_c, M_c, L_c, tau_c])

    def get_state_derivative(self, r, state, return_kappa=False):
        """
        Calculates the elementwise derivative of the state vector.

        :param r: The current radius.
        :param state: The state vector at the given radius.
        :param return_kappa: If set to True, then the opacity will be returned as the second item of a tuple.
        :return: The elementwise derivative of the state vector, and optionally the optical depth as well.
        """
        rho, T, M, L, _ = state
        kappa_value = self.kappa(rho, T)

        T_prime_value = self.T_prime(r, rho, T, M, L, kappa_value=kappa_value)
        rho_prime_value = self.rho_prime(r, rho, T, M, L, T_prime_value=T_prime_value)
        M_prime_value = self.M_prime(r, rho)
        L_prime_value = self.L_prime(r, rho, T, M_prime_value=M_prime_value)
        tau_prime_value = self.tau_prime(rho, T, kappa_value=kappa_value)

        state_derivative = np.array([rho_prime_value, T_prime_value, M_prime_value, L_prime_value, tau_prime_value])
        return (state_derivative, kappa_value) if return_kappa else state_derivative

    def rho_prime(self, r, rho, T, M, L, T_prime_value=None):
        """
        The T_prime_value can optionally be provided to prevent redoing the calculation.
        """
        if T_prime_value is None:
            T_prime_value = self.T_prime(r, rho, T, M, L)
        return -(G * M * rho / r ** 2 * (1 + self.lambda_small / r) * (1 + r / self.lambda_large) +
                 self.dP_by_dT(rho, T) * T_prime_value) / self.dP_by_drho(rho, T)

    def T_prime(self, r, rho, T, M, L, kappa_value=None):
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        radiative = self.T_prime_radiative(r, rho, T, L, kappa_value=kappa_value)
        convective = self.T_prime_convective(r, rho, T, M)
        return np.maximum(radiative, convective)

    def T_prime_radiative(self, r, rho, T, L, kappa_value=None):
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        return -3 * kappa_value * rho * L / (16 * pi * a * c * T ** 3 * r ** 2)

    def T_prime_convective(self, r, rho, T, M):
        return -(1 - 1 / gamma) * T * G * M * rho / (self.P(rho, T) * r ** 2) * (1 + self.lambda_small / r) * (
                    1 + r / self.lambda_large)

    def is_convective(self, r, rho, T, M, L, kappa_value=None):
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        return self.T_prime_convective(r, rho, T, M) > self.T_prime_radiative(r, rho, T, L, kappa_value=kappa_value)

    @staticmethod
    def M_prime(r, rho):
        return 4 * pi * r ** 2 * rho

    def L_prime(self, r, rho, T, M_prime_value=None):
        if M_prime_value is None:
            M_prime_value = self.M_prime(r, rho)
        return M_prime_value * self.epsilon(rho, T)

    def L_proton_proton_prime(self, r, rho, T, M_prime_value=None):
        if M_prime_value is None:
            M_prime_value = self.M_prime(r, rho)
        return M_prime_value * self.epsilon_proton_proton(rho, T)

    def L_CNO_prime(self, r, rho, T, M_prime_value=None):
        if M_prime_value is None:
            M_prime_value = self.M_prime(r, rho)
        return M_prime_value * self.epsilon_CNO(rho, T)

    def tau_prime(self, rho, T, kappa_value=None):
        if kappa_value is None:
            kappa_value = self.kappa(rho, T)
        return kappa_value * rho

    def P(self, rho, T):
        return self.P_degeneracy(rho) + self.P_gas(rho, T) + self.P_photon(T)

    @staticmethod
    def P_degeneracy(rho):
        if rho < 0:
            return 1e-10
        return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (5 * m_e) * (rho / m_p) ** (5 / 3)

    def P_gas(self, rho, T):
        return rho * k_b * T / (self.mu() * m_p)

    @staticmethod
    def P_photon(T):
        return (1 / 3) * a * T ** 4

    def dP_by_drho(self, rho, T):
        if rho < 0:
            return 1e-10
        return (3 * pi ** 2) ** (2 / 3) * h_bar ** 2 / (3 * m_e * m_p) * (rho / m_p) ** (2 / 3) + k_b * T / (
                self.mu() * m_p)

    def dP_by_dT(self, rho, T):
        return rho * k_b / (self.mu() * m_p) + (4 / 3) * a * T ** 3

    def kappa(self, rho, T):
        return (1 / self.kappa_H_minus(rho, T) + 1 / np.maximum(self.kappa_es(), self.kappa_ff(rho, T))) ** -1

    def kappa_es(self):
        return kappa_es_coefficient * (self.X + 1)

    def kappa_ff(self, rho, T):
        if rho < 0:
            return 1e-10
        return kappa_ff_coefficient * (self.Z + 0.0001) * (rho / 10 ** 3) ** 0.7 * T ** -3.5

    def kappa_H_minus(self, rho, T):
        if rho < 0:
            return 1e-10
        return kappa_H_minus_coefficient * (self.Z / 0.02) * (rho / 10 ** 3) ** 0.5 * T ** 9

    def epsilon(self, rho, T):
        return self.epsilon_proton_proton(rho, T) + self.epsilon_CNO(rho, T)

    def epsilon_proton_proton(self, rho, T):
        return epsilon_proton_proton_coefficient * self.X ** 2 * (rho / 10 ** 5) * (T / 10 ** 6) ** 4

    def epsilon_CNO(self, rho, T, X_CNO=None):
        if X_CNO is None:
            X_CNO = 0.03 * self.X
        return epsilon_CNO_coefficient * self.X * X_CNO * (rho / 10 ** 5) * (T / 10 ** 6) ** 19.9

    def mu(self):
        return (2 * self.X + 0.75 * self.Y + 0.5 * self.Z) ** -1

    def __str__(self):
        return 'StellarConfiguration(X=' + str(self.X) + ',Y=' + str(self.Y) + ',Z=' + str(self.Z) + \
               ',lambda_small=' + str(self.lambda_small / R_sun) + \
               'R_sun,lambda_large=' + str(self.lambda_small / R_sun) + 'R_sun)'

    def has_gravity_modifications(self):
        return self.lambda_small != lambda_small_default or self.lambda_large != lambda_large_default

    def describe_gravity_modifications(self):
        description = ''
        if self.lambda_small != lambda_small_default:
            description += r'$\lambda$=' + ('%.2f' % (self.lambda_small / R_sun)) + r'$R_{\odot}$'
        if self.lambda_large != lambda_large_default:
            description += r'$\Lambda$=' + ('%.2f' % (self.lambda_large / R_sun)) + r'$R_{\odot}$'
        return description

