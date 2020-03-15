# noinspection PyUnresolvedReferences
from numpy import pi  # Added so pi can be used in other files
import numpy as np

G = 6.67408e-11  # m^3/(kgs^2)
sigma = 5.670374419e-8  # W/(m^2K^4)
a = 7.56591e-16  # J/(m^3K^4)
c = 299_792_458  # m/s
m_p = 1.6726219e-27  # kg
m_e = 9.10938356e-31  # kg
h_bar = 1.054571817e-34  # Js
k_b = 1.38064852e-23  # kgm^2/(Ks^2)

gamma = 5 / 3  # unitless

kappa_es_coefficient = 0.02  # m^2/kg
kappa_ff_coefficient = 1.0e24  # m^2/kg
kappa_H_minus_coefficient = 2.5e-32  # m^2/kg

epsilon_proton_proton_coefficient = 1.07e-7  # W/kg
epsilon_CNO_coefficient = 8.24e-26  # W/kg

M_sun = 1.98847e30  # kg
R_sun = 696_340_000  # m
L_sun = 3.828e26  # W

# Divide a state vector by this to get it in units of g/cm^3, K, M_sun, L_sun, 1
state_normalize_vector = np.array([10 ** 3, 1, M_sun, L_sun, 1])
