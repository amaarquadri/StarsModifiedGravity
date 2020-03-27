import numpy as np

# Base SI Units
kg = 1  # kg
m = 1  # m
s = 1  # s
K = 1  # K

# Derived Standard SI units
J = kg * m ** 2 / s ** 2  # J
W = J / s  # W

# Non Standard Units
g = 0.001 * kg  # kg
cm = 0.01 * m  # m
erg = 1E-7 * J  # J
million_K = 1E6 * K  # K

# Astronomical Data/Units
M_sun = 1.98847e30 * kg  # kg
L_sun = 3.828e26 * W  # W
R_sun = 696_340_000 * m  # m

# For each of the following units vectors:
# Multiplying a vector by the units vector converts from the shown units into standard units
# Dividing a vector by the units vectors converts from standard units into the shown units

state_vector_units = np.array([g / cm ** 3, K, M_sun, L_sun, 1])
state_prime_vector_units = state_vector_units / R_sun

state_test_vector_units = np.array([58.56 * g / cm ** 3, 8.23e6 * K, 0.673 * M_sun, 5.86e-2 * L_sun, 1])
state_prime_test_vector_units = state_test_vector_units / (0.865 * R_sun)

example_star_units = np.array([g,
                               cm,
                               g / cm ** 3,
                               K,
                               g,
                               erg / s,
                               erg / s / cm, erg / s / cm, erg / s / cm,
                               1, 1, 1, 1, 1, 1,
                               cm ** 2 / g, cm ** 2 / g, cm ** 2 / g, cm ** 2 / g,
                               erg / cm ** 3, erg / cm ** 3, erg / cm ** 3, erg / cm ** 3,
                               g / cm ** 4,
                               K / cm,
                               g / cm,
                               erg / s / cm])
