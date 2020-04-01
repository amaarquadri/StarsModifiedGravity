import numpy as np
import matplotlib.pyplot as plt

from src.NumericalIntegration import rho_index, T_index, L_index, solve_bvp
from src.Units import g, cm, million_K
from src.StarSequenceGenerator import generate_stars, L_surface_index, T_surface_index

'''
Module to generate an HR Plot
'''


def hr_plot(temp_data, lumin_data):
    """
    """

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
    # ax1.set_ylim(min_ylim, max_ylim)

    ax2 = ax1.twiny()

    ax2.set_xlabel('Stellar Class', labelpad=10)
    ax2.tick_params(axis='x', labelsize=0, labeltop=True)
    ax2.set_xscale("log")
    ax2.set_xlim((max_xlim, min_xlim))
    ax2.plot()  # keep empty

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
    """
    """

    num_space = np.linspace(0.01, 36, 5)
    core_temp_range = [int(num * million_K) for num in num_space]

    all_lumin = []
    all_temp = []
    last_rho_c = 100 * g / cm ** 3
    for core_temp in core_temp_range:
        try:
            r_values, state_values = solve_bvp(core_temp, last_rho_c)
            last_rho_c = state_values[rho_index, 0]
            surf_temp, surf_lumin = state_values[T_index, -1], state_values[L_index, -1]

            if surf_temp < 1e5: # anything above this is an outlier
                all_lumin.append(surf_lumin)
                all_temp.append(surf_temp)
        except ValueError:
            print("'solve_numerically' didn't like Core Temp = {}.".format(core_temp))

    hr_plot(all_temp, all_lumin)


if __name__ == '__main__':
    stellar_data = generate_stars(np.linspace(0.01 * million_K, 36 * million_K, 20))
    hr_plot(stellar_data[T_surface_index, :], stellar_data[L_surface_index, :])
