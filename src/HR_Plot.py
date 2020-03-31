import numpy as np
import matplotlib.pyplot as plt
from Units import K, g, cm, million_K, M_sun, L_sun, R_sun
from Utils import find_zeros_index, interpolate
from StellarStructureEquations import L_proton_proton_prime, L_CNO_prime, P_degeneracy, P_gas, P_photon,\
    kappa, kappa_es, kappa_ff, kappa_H_minus
from NumericalIntegration import rho_index, T_index, M_index, L_index, tau_index, solve_numerically
from ExampleStar import ex_r_index, ex_rho_index, ex_T_index, ex_M_index, ex_L_index, \
    ex_P_index, ex_P_degeneracy_index, ex_P_gas_index, ex_P_photon_index, \
    ex_kappa_index, ex_kappa_es_index, ex_kappa_ff_index, ex_kappa_H_minus_index, \
    ex_L_prime_index, ex_L_proton_proton_prime_index, ex_L_CNO_prime_index, \
    ex_dlog_P_by_dlog_T_index

'''
Module to generate an HR Plot
'''

def get_luminosity(r_values, state_values):
    '''
    '''

    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index] - 2 / 3)

    surface_state = interpolate(state_values, surface_index)
    surface_L = surface_state[L_index]
    surface_T = surface_state[T_index]

    L_surf = surface_L/L_sun
    T_surf = surface_T/K
    
    return T_surf, L_surf

def hr_plot(temp_data, lumin_data):
    '''
    '''

    # insert data here
    xdata = np.array(temp_data)
    ydata = np.array(lumin_data)

    # find limits

    max_x = max(xdata)
    min_x = min(xdata)

    max_y = max(ydata)
    min_y = min(ydata)

    log_max_x = np.log10(max_x)
    log_min_x = np.log10(min_x)

    log_max_y = np.log10(max_y)
    log_min_y = np.log10(min_y)

    new_min_values = []
    for value in [log_min_x, log_min_y]:
        if value < 0:
            new_value = int(value) - 1
        elif value == 0:
            new_value = int(value)
        else:
            new_value = int(value)
        new_min_values.append(new_value)

    new_max_values = []
    for value in [log_max_x, log_max_y]:
        if value < 0:
            new_value = int(value)
        elif value == 0:
            new_value = int(value)
        else:
            new_value = int(value) + 1
        new_max_values.append(new_value)

    min_xlim = 10**(new_min_values[0]) # min x limit
    max_xlim = 10**(new_max_values[0]) # max x limit

    min_ylim = 10**(new_min_values[1]) # minimal y limit
    max_ylim = 10**(new_max_values[1]) # maximal y limit

    # Plotting

    fig, ax1 = plt.subplots(constrained_layout=True)

    fig.suptitle("H-R Diagram")
    ax1.set_xlabel("Surface Temperature (in Kelvin)")
    ax1.set_ylabel("Solar Luminosity ($L/L_o$)", labelpad=18)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(xdata, ydata, "o")
    ax1.set_xlim((max_xlim, min_xlim))
    ax1.set_ylim(min_ylim, max_ylim)

    ax2 = ax1.twiny()

    ax2.set_xlabel('Stellar Class', labelpad=10)
    ax2.tick_params(axis='x', labelsize=0, labeltop=True)
    ax2.set_xscale("log")
    ax2.set_xlim((max_xlim, min_xlim))
    ax2.plot() # keep empty

    all_annotations = [3e4, 17320, 8660, 6708, 5585, 4386, 2980]
    all_stellar_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

    valid_annotations = []
    valid_stellar_classes = []
    for index, annotation in enumerate(all_annotations):
        if min_x <= annotation <= max_x:
            valid_annotations.append(annotation)
            valid_stellar_classes.append(all_stellar_classes[index])

    for index, annotation in enumerate(valid_annotations):
        ax1.annotate(
            valid_stellar_classes[index],
            xy=(annotation, 298),
            xycoords=("data", "figure points"),
            horizontalalignment="center",
            verticalalignment="bottom"
        )
    plt.show()

def generate_hr():
    '''
    '''

    num_space = np.linspace(0.01, 36, 5)
    core_temp_range = [int(num * million_K) for num in num_space]

    all_lumin = []
    all_temp = []
    for core_temp in core_temp_range:
        try:
            r_values, state_values = solve_numerically(core_temp)
            surf_temp, surf_lumin = get_luminosity(r_values, state_values)

            if surf_temp < 1e5: # anything above this is an outlier
                all_lumin.append(surf_lumin)
                all_temp.append(surf_temp)
        except ValueError:
            print("'solve_numerically' didn't like Core Temp = {}.".format(core_temp))

    hr_plot(all_temp, all_lumin)