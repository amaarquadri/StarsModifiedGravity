import numpy as np
import matplotlib.pyplot as plt
from src.Units import K, g, cm, million_K, M_sun, L_sun, R_sun
from src.Utils import find_zeros_index, interpolate
from src.StellarStructureEquations import L_proton_proton_prime, L_CNO_prime, P_degeneracy, P_gas, P_photon,\
    kappa, kappa_es, kappa_ff, kappa_H_minus
from src.NumericalIntegration import rho_index, T_index, M_index, L_index, tau_index
from src.ExampleStar import ex_r_index, ex_rho_index, ex_T_index, ex_M_index, ex_L_index, \
    ex_P_index, ex_P_degeneracy_index, ex_P_gas_index, ex_P_photon_index, \
    ex_kappa_index, ex_kappa_es_index, ex_kappa_ff_index, ex_kappa_H_minus_index, \
    ex_L_prime_index, ex_L_proton_proton_prime_index, ex_L_CNO_prime_index, \
    ex_dlog_P_by_dlog_T_index


def graph_star(r_values, state_values, name="Sun", reference_data=None):
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_state = interpolate(state_values, surface_index)
    rho_c = state_values[rho_index, 0]
    T_c = state_values[T_index, 0]
    surface_M = surface_state[M_index]
    surface_L = surface_state[L_index]
    surface_T = surface_state[T_index]

    print("Central Density:", rho_c / (g / cm ** 3), r"$\frac{g}{cm^3}$")
    print("Central Temperature:", T_c / million_K, "million K")
    print("Radius:", surface_r / R_sun, r"$R_{sun}$")
    print("Mass:", state_values[M_index, -1] / M_sun, r"$M_{sun}$")
    print("Luminosity:", surface_L / L_sun, r"$L_{sun}$")
    print("Surface Temperature:", surface_T / K, "K")

    surface_index = int(surface_index)
    r_graph_values = r_values[:surface_index] / surface_r

    plt.plot(r_graph_values, state_values[rho_index, :surface_index] / rho_c, label=r"$\rho$", color="black")
    plt.plot(r_graph_values, state_values[T_index, :surface_index] / T_c, label="T", color="red")
    plt.plot(r_graph_values, state_values[M_index, :surface_index] / surface_M, label="M", color="green")
    plt.plot(r_graph_values, state_values[L_index, :surface_index] / surface_L, label="L", color="blue")
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / reference_data[ex_r_index, -1]
        plt.plot(r_ref_values, reference_data[ex_rho_index, :] / reference_data[ex_rho_index, 0],
                 label=r"$\rho_{ref}$", color="black", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_T_index, :] / reference_data[ex_T_index, 0],
                 label=r"$T_{ref}$", color="red", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_M_index, :] / reference_data[ex_M_index, -1],
                 label=r"$M_{ref}$", color="green", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_L_index, :] / reference_data[ex_L_index, -1],
                 label=r"$L_{ref}$", color="blue", linestyle='dashed')

    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Properties Versus Radius Fraction")
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{\rho}{\rho_{c}}, \frac{T}{T_{c}}, \frac{M}{M_{star}}, \frac{L}{L_{star}}$")
    plt.savefig("../Graphs/" + name + "_properties.png")
    plt.clf()

    P_degeneracy_values = P_degeneracy(state_values[rho_index, :surface_index])
    P_gas_values = P_gas(state_values[rho_index, :surface_index], state_values[T_index, :surface_index])
    P_photon_values = P_photon(state_values[T_index, :surface_index])
    P_total_values = P_degeneracy_values + P_gas_values + P_photon_values
    P_max = P_total_values[0]
    plt.plot(r_graph_values, P_total_values / P_max, label=r"$P_{total}$", color="black")
    plt.plot(r_graph_values, P_degeneracy_values / P_max, label=r"$P_{deg}$", color="red")
    plt.plot(r_graph_values, P_gas_values / P_max, label=r"$P_{gas}$", color="green")
    plt.plot(r_graph_values, P_photon_values / P_max, label=r"$P_{photon}$", color="blue")
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / reference_data[ex_r_index, -1]
        P_ref_max = reference_data[ex_P_index, 0]
        plt.plot(r_ref_values, reference_data[ex_P_index, :] / P_ref_max,
                 label=r"$P_{total, ref}$", color="black", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_P_degeneracy_index, :] / P_ref_max,
                 label=r"$P_{deg, ref}$", color="red", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_P_gas_index, :] / P_ref_max,
                 label=r"$P_{gas, ref}$", color="green", linestyle='dashed')
        plt.plot(r_ref_values, reference_data[ex_P_photon_index, :] / P_ref_max,
                 label=r"$P_{photon, ref}$", color="blue", linestyle='dashed')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{P}{P_c}$")
    plt.title("Pressure Contributions Versus Radius Fraction")
    plt.savefig("../Graphs/" + name + "_pressure.png")
    plt.clf()

    kappa_es_values = kappa_es(state_values[rho_index, :surface_index])
    kappa_ff_values = kappa_ff(state_values[rho_index, :surface_index], state_values[T_index, :surface_index])
    kappa_H_minus_values = kappa_H_minus(state_values[rho_index, :surface_index], state_values[T_index, :surface_index])
    kappa_total_values = kappa(state_values[rho_index, :surface_index], state_values[T_index, :surface_index])
    plt.plot(r_graph_values, np.log10(kappa_total_values / (cm ** 2 / g)), label=r"$\kappa_{total}$", color="black")
    plt.plot(r_graph_values, np.log10(kappa_es_values / (cm ** 2 / g)), label=r"$\kappa_{es}$", color="blue")
    plt.plot(r_graph_values, np.log10(kappa_ff_values / (cm ** 2 / g)), label=r"$\kappa_{ff}$", color="green")
    plt.plot(r_graph_values, np.log10(kappa_H_minus_values / (cm ** 2 / g)), label=r"$\kappa_{H-}$", color="red")
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / reference_data[ex_r_index, -1]
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
    plt.title("Opacity Contributions Versus Radius Fraction")
    plt.savefig("../Graphs/" + name + "_opacity.png")
    plt.clf()

    L_proton_proton_prime_values = L_proton_proton_prime(r_values[:surface_index],
                                                         state_values[rho_index, :surface_index],
                                                         state_values[T_index, :surface_index])
    L_CNO_prime_values = L_CNO_prime(r_values[:surface_index], state_values[rho_index, :surface_index],
                                     state_values[T_index, :surface_index])
    L_total_prime = L_proton_proton_prime_values + L_CNO_prime_values
    plt.plot(r_graph_values, L_total_prime / surface_L * surface_r, label=r"$\frac{L}{dr}$", color="black")
    plt.plot(r_graph_values, L_proton_proton_prime_values / surface_L * surface_r,
             label=r"$\frac{L_{proton-proton}}{dr}$", color="red")
    plt.plot(r_graph_values, L_CNO_prime_values / surface_L * surface_r, label=r"$\frac{L_{CNO}}{dr}$", color="blue")
    if reference_data is not None:
        r_ref_values = reference_data[ex_r_index, :] / reference_data[ex_r_index, -1]
        ref_L_surface = reference_data[ex_L_index, -1]
        ref_r_surface = reference_data[ex_r_index, -1]
        plt.plot(r_ref_values, reference_data[ex_L_prime_index, :] / ref_L_surface * ref_r_surface,
                 label=r"$\frac{L_{ref}}{dr}$", color="black", linestyle="dashed")
        plt.plot(r_ref_values, reference_data[ex_L_proton_proton_prime_index, :] / ref_L_surface * ref_r_surface,
                 label=r"$\frac{L_{proton-proton, ref}}{dr}$", color="red", linestyle="dashed")
        plt.plot(r_ref_values, reference_data[ex_L_CNO_prime_index, :] / ref_L_surface * ref_r_surface,
                 label=r"$\frac{L_{CNO, ref}}{dr}$", color="blue", linestyle="dashed")
    plt.legend()
    plt.xlim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{dL}{dr} (\frac{L_{star}}{R_{star}})$")
    plt.title("Luminosity Gradient Contributions Versus Radius Fraction")
    plt.savefig("../Graphs/" + name + "_luminosity.png")
    plt.clf()

    log_P_values = np.log(P_total_values)
    log_T_values = np.log(state_values[T_index, :surface_index])
    dlogP_dlogT_values = np.diff(log_P_values) / np.diff(log_T_values)
    # Omit last r_graph_value since taking first difference decreases array size by 1
    plt.plot(r_graph_values[:-1], dlogP_dlogT_values, label="calculated", color="black")
    if reference_data is not None:
        plt.plot(reference_data[ex_r_index, :] / reference_data[ex_r_index, -1],
                 reference_data[ex_dlog_P_by_dlog_T_index, :],
                 label="ref", color="black", linestyle="dashed")
    plt.xlim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{dlogP}{dlogT}$")
    plt.title(r"$\frac{dlogP}{dlogT}$ Versus Radius Fraction")
    plt.savefig("../Graphs/" + name + "_dlogP_dlogT.png")
    plt.clf()
