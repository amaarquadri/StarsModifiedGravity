import matplotlib.pyplot as plt
import numpy as np

from src.example_star import ex_r_index, ex_rho_index, ex_T_index, ex_M_index, ex_L_index, \
    ex_P_index, ex_P_degeneracy_index, ex_P_gas_index, ex_P_photon_index, \
    ex_kappa_index, ex_kappa_es_index, ex_kappa_ff_index, ex_kappa_H_minus_index, \
    ex_L_prime_index, ex_L_proton_proton_prime_index, ex_L_CNO_prime_index, \
    ex_dlog_P_by_dlog_T_index
from src.file_io import get_timestamp
from src.stellar_structure_equations import StellarConfiguration, rho_index, T_index, M_index, L_index
from src.units import K, g, cm, million_K, M_sun, L_sun, R_sun


def graph_star(r_values, state_values, name=None, reference_data=None, config=StellarConfiguration()):
    """

    :param r_values: The radius values of this star.
    :param state_values: The corresponding state matrix of this star.
    :param name: The file name prefix for this star's graphs to be saved with.
    :param reference_data: If provided, it will be graphed with dashed lines with the same scale as the stellar data.
    :param config: The stellar configuration that was used to generate this star.
    """
    if name is None:
        name = 'test_star_' + get_timestamp()

    title_description = ' (' + config.describe_gravity_modifications() + ')' \
        if config.has_gravity_modifications() else ''

    surface_r = r_values[-1]
    surface_state = state_values[:, -1]
    rho_c = state_values[rho_index, 0]
    T_c = state_values[T_index, 0]
    surface_M = surface_state[M_index]
    surface_L = surface_state[L_index]
    surface_T = surface_state[T_index]

    print("Central Density:", rho_c / (g / cm ** 3), r"$\frac{g}{cm^3}$")
    print("Central Temperature:", T_c / million_K, "million K")
    print("Radius:", surface_r / R_sun, r"$R_{sun}$")
    print("Mass:", surface_M / M_sun, r"$M_{sun}$")
    print("Luminosity:", surface_L / L_sun, r"$L_{sun}$")
    print("Surface Temperature:", surface_T / K, "K")

    r_graph_values = r_values / surface_r

    # Compute convective regions
    kappa_total_values = config.kappa(state_values[rho_index, :], state_values[T_index, :])
    convective_regions = get_convective_regions(r_values, state_values, surface_r, kappa_total_values, config=config)

    def mark_convective_regions(facecolor='gray', alpha=0.2):
        for a, b in convective_regions:
            plt.axvspan(a, b, facecolor=facecolor, alpha=alpha)

    plt.plot(r_graph_values, state_values[rho_index, :] / rho_c, label=r"$\rho$", color="black")
    plt.plot(r_graph_values, state_values[T_index, :] / T_c, label="T", color="red")
    plt.plot(r_graph_values, state_values[M_index, :] / surface_M, label="M", color="green")
    plt.plot(r_graph_values, state_values[L_index, :] / surface_L, label="L", color="blue")
    mark_convective_regions()
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / surface_r
        plt.plot(r_ref_values, reference_data[ex_rho_index, :] / rho_c,
                 label=r"$\rho_{ref}$", color="black", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_T_index, :] / T_c,
                 label=r"$T_{ref}$", color="red", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_M_index, :] / surface_M,
                 label=r"$M_{ref}$", color="green", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_L_index, :] / surface_L,
                 label=r"$L_{ref}$", color="blue", linestyle='dashed')

    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(r"Properties Versus Radius Fraction" + title_description)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{\rho}{\rho_{c}}, \frac{T}{T_{c}}, \frac{M}{M_{star}}, \frac{L}{L_{star}}$")
    plt.savefig("../Graphs/" + name + "_properties.png")
    plt.clf()

    P_degeneracy_values = config.P_degeneracy(state_values[rho_index, :])
    P_gas_values = config.P_gas(state_values[rho_index, :], state_values[T_index, :])
    P_photon_values = config.P_photon(state_values[T_index, :])
    P_total_values = P_degeneracy_values + P_gas_values + P_photon_values
    P_max = P_total_values[0]
    plt.plot(r_graph_values, P_total_values / P_max, label=r"$P_{total}$", color="black")
    plt.plot(r_graph_values, P_degeneracy_values / P_max, label=r"$P_{deg}$", color="red")
    plt.plot(r_graph_values, P_gas_values / P_max, label=r"$P_{gas}$", color="green")
    plt.plot(r_graph_values, P_photon_values / P_max, label=r"$P_{photon}$", color="blue")
    mark_convective_regions()
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / surface_r
        plt.plot(r_ref_values, reference_data[ex_P_index, :] / P_max,
                 label=r"$P_{total, ref}$", color="black", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_P_degeneracy_index, :] / P_max,
                 label=r"$P_{deg, ref}$", color="red", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_P_gas_index, :] / P_max,
                 label=r"$P_{gas, ref}$", color="green", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_P_photon_index, :] / P_max,
                 label=r"$P_{photon, ref}$", color="blue", linestyle='dashed')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{P}{P_c}$")
    plt.title(r"Pressure Contributions Versus Radius Fraction" + title_description)
    plt.savefig("../Graphs/" + name + "_pressure.png")
    plt.clf()

    kappa_es_values = config.kappa_es() * np.ones(len(r_graph_values))
    kappa_ff_values = config.kappa_ff(state_values[rho_index, :], state_values[T_index, :])
    kappa_H_minus_values = config.kappa_H_minus(state_values[rho_index, :], state_values[T_index, :])
    plt.plot(r_graph_values, np.log10(kappa_total_values / (cm ** 2 / g)), label=r"$\kappa_{total}$", color="black")
    plt.plot(r_graph_values, np.log10(kappa_es_values / (cm ** 2 / g)), label=r"$\kappa_{es}$", color="blue")
    plt.plot(r_graph_values, np.log10(kappa_ff_values / (cm ** 2 / g)), label=r"$\kappa_{ff}$", color="green")
    plt.plot(r_graph_values, np.log10(kappa_H_minus_values / (cm ** 2 / g)), label=r"$\kappa_{H-}$", color="red")
    mark_convective_regions()
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / surface_r
        plt.plot(r_ref_values, np.log10(reference_data[ex_kappa_index, :] / (cm ** 2 / g)),
                 label=r"$\kappa_{total, ref}$", color="black", linestyle='dashed')
        plt.plot(r_ref_values, np.log10(reference_data[ex_kappa_es_index, :] / (cm ** 2 / g)),
                 label=r"$\kappa_{es, ref}$", color="blue", linestyle='dashed')
        plt.plot(r_ref_values, np.log10(reference_data[ex_kappa_ff_index, :] / (cm ** 2 / g)),
                 label=r"$\kappa_{ff, ref}$", color="green", linestyle='dashed')
        plt.plot(r_ref_values, np.log10(reference_data[ex_kappa_H_minus_index, :] / (cm ** 2 / g)),
                 label=r"$\kappa_{H-, ref}$", color="red", linestyle='dashed')
    plt.legend()
    plt.xlim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\log_{10}(\kappa) (\frac{cm^2}{g})$")
    plt.title(r"Opacity Contributions Versus Radius Fraction" + title_description)
    plt.savefig("../Graphs/" + name + "_opacity.png")
    plt.clf()

    L_proton_proton_prime_values = config.L_proton_proton_prime(r_values,
                                                                state_values[rho_index, :],
                                                                state_values[T_index, :])
    L_CNO_prime_values = config.L_CNO_prime(r_values, state_values[rho_index, :],
                                            state_values[T_index, :])
    L_total_prime = L_proton_proton_prime_values + L_CNO_prime_values
    plt.plot(r_graph_values, L_total_prime / surface_L * surface_r, label=r"$\frac{L}{dr}$", color="black")
    plt.plot(r_graph_values, L_proton_proton_prime_values / surface_L * surface_r,
             label=r"$\frac{L_{proton-proton}}{dr}$", color="red")
    plt.plot(r_graph_values, L_CNO_prime_values / surface_L * surface_r, label=r"$\frac{L_{CNO}}{dr}$", color="blue")
    mark_convective_regions()
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / surface_r
        plt.plot(r_ref_values, reference_data[ex_L_prime_index, :] / surface_L * surface_r,
                 label=r"$\frac{L_{ref}}{dr}$", color="black", linestyle="dashed")
        plt.plot(r_ref_values, reference_data[ex_L_proton_proton_prime_index, :] / surface_L * surface_r,
                 label=r"$\frac{L_{proton-proton, ref}}{dr}$", color="red", linestyle="dashed")
        plt.plot(r_ref_values, reference_data[ex_L_CNO_prime_index, :] / surface_L * surface_r,
                 label=r"$\frac{L_{CNO, ref}}{dr}$", color="blue", linestyle="dashed")
    plt.legend()
    plt.xlim(0, 1)
    plt.title(r"Luminosity Gradient Contributions Versus Radius Fraction" + title_description)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{dL}{dr} (\frac{L_{star}}{R_{star}})$")
    plt.savefig("../Graphs/" + name + "_luminosity.png")
    plt.clf()

    log_P_values = np.log(P_total_values)
    log_T_values = np.log(state_values[T_index, :])
    dlogP_dlogT_values = np.diff(log_P_values) / np.diff(log_T_values)  # can result in Nans
    # Omit last 2 r_graph_values since taking first difference decreases array size by 1, and last point is unstable
    plt.plot(r_graph_values[:-2], dlogP_dlogT_values[:-1], label="calculated", color="black")
    mark_convective_regions()
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / surface_r
        plt.plot(r_ref_values, reference_data[ex_dlog_P_by_dlog_T_index, :],
                 label="ref", color="black", linestyle="dashed")
    plt.xlim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{dlogP}{dlogT}$")
    plt.title(r"$\frac{dlogP}{dlogT}$ Versus Radius Fraction" + title_description)
    plt.savefig("../Graphs/" + name + "_dlogP_dlogT.png")
    plt.clf()


def get_convective_regions(r_values, state_values, surface_r, kappa_total_values=None, config=StellarConfiguration()):
    if kappa_total_values is None:
        kappa_total_values = config.kappa(state_values[rho_index, :], state_values[T_index, :])
    convective = config.is_convective(r_values, state_values[rho_index, :],
                                      state_values[T_index, :], state_values[M_index, :],
                                      state_values[L_index, :], kappa_value=kappa_total_values)
    indices = iter(np.concatenate(([0], np.argwhere(np.diff(convective))[:, 0], [len(convective) - 1])))
    if not convective[0]:
        next(indices)  # remove first
    pairs = []
    while True:
        try:
            a = next(indices)
            b = next(indices)
            pairs.append((r_values[a] / surface_r, r_values[b] / surface_r))
        except StopIteration:
            return pairs
