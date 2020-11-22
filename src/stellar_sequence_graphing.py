import matplotlib.pyplot as plt
import numpy as np

from src.file_io import get_timestamp, load_stellar_data
from src.star_sequence_generator import T_surface_index, L_surface_index, M_surface_index, R_index
from src.units import L_sun, K, M_sun, R_sun


def plot_sequence(stellar_data_lists, file_name=None):
    if file_name is None:
        file_name = 'test' + get_timestamp()

    # Remove stars that didnt converge
    stellar_data_lists = [(stellar_data[:, stellar_data[M_surface_index, :] < 1000 * M_sun], config)
                          for stellar_data, config in stellar_data_lists]

    plot_HR(stellar_data_lists, file_name + '_HR')
    plot_LM(stellar_data_lists, file_name + '_LM')
    plot_RM(stellar_data_lists, file_name + '_RM')


def plot_HR(stellar_data_lists, file_name=None):
    if file_name is None:
        file_name = 'HR_plot_' + get_timestamp()

    for stellar_data, config in stellar_data_lists:
        label = config.describe_gravity_modifications() if config.has_gravity_modifications() else 'standard'
        plt.scatter(np.log10(stellar_data[T_surface_index, :] / K),
                    np.log10(stellar_data[L_surface_index, :] / L_sun), label=label)
    if len(stellar_data_lists) > 1:
        plt.legend()
    plt.title('HR Diagram')
    plt.xlabel('$Log_{10}$ of Temperature ($Log_{10}$ (K))')
    plt.ylabel(r'$Log_{10}$ of Luminosity ($Log_{10}$ ($L_{\odot}$))')
    plt.gca().invert_xaxis()
    plt.savefig('../Graphs/' + file_name + '.png')
    plt.clf()


def plot_LM(stellar_data_lists, file_name=None):
    if file_name is None:
        file_name = 'LM_plot_' + get_timestamp()

    for stellar_data, config in stellar_data_lists:
        label = config.describe_gravity_modifications() if config.has_gravity_modifications() else 'standard'
        plt.scatter(np.log10(stellar_data[M_surface_index, :] / M_sun),
                    np.log10(stellar_data[L_surface_index, :] / L_sun), label=label)
        line = np.polyfit(np.log10(stellar_data[M_surface_index, :] / M_sun),
                          np.log10(stellar_data[L_surface_index, :] / L_sun), deg=1)
        print('LM Line of best fit for', label, ': y=', line[0], '* x +', line[1])
    if len(stellar_data_lists) > 1:
        plt.legend()
    plt.title('LM Diagram')
    plt.xlabel(r'$Log_{10}$ of Mass ($Log_{10}$ ($M_{\odot}$))')
    plt.ylabel(r'$Log_{10}$ of Luminosity ($Log_{10}$ ($L_{\odot}$))')
    plt.savefig('../Graphs/' + file_name + '.png')
    plt.clf()


def plot_RM(stellar_data_lists, file_name=None):
    if file_name is None:
        file_name = 'RM_plot_' + get_timestamp()

    for stellar_data, config in stellar_data_lists:
        label = config.describe_gravity_modifications() if config.has_gravity_modifications() else 'standard'
        plt.scatter(np.log10(stellar_data[M_surface_index, :] / M_sun),
                    np.log10(stellar_data[R_index, :] / R_sun), label=label)
        line = np.polyfit(np.log10(stellar_data[M_surface_index, :] / M_sun),
                          np.log10(stellar_data[R_index, :] / L_sun), deg=1)
        print('RM Line of best fit for', label, ': y=', line[0], '* x +', line[1])
    if len(stellar_data_lists) > 1:
        plt.legend()
    plt.title('RM Diagram')
    plt.xlabel(r'$Log_{10}$ of Mass ($Log_{10}$ ($M_{\odot}$))')
    plt.ylabel(r'$Log_{10}$ of Radius ($Log_{10}$ ($R_{\odot}$))')
    plt.savefig('../Graphs/' + file_name + '.png')
    plt.clf()


if __name__ == '__main__':
    file_names = ['standard_stellar_data']
    dataset = [load_stellar_data(file_name=file_name, return_config=True) for file_name in file_names]
    plot_sequence(dataset)
