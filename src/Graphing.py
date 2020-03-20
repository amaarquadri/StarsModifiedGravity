from src.Utils import find_zeros_index, interpolate
from src.NumericalIntegration import rho_index, T_index, M_index, L_index, tau_index
from src.StellarStructureEquations import *
import matplotlib.pyplot as plt


def graph_star(r_values, state_values, name="Sun"):
    tau_infinity = state_values[tau_index, -1]
    surface_index = find_zeros_index(tau_infinity - state_values[tau_index] - 2 / 3)

    surface_r = interpolate(r_values, surface_index)
    surface_state = interpolate(state_values, surface_index)
    rho_c = state_values[rho_index, 0]
    T_c = state_values[T_index, 0]
    surface_M = surface_state[M_index]
    surface_L = surface_state[L_index]
    surface_T = surface_state[T_index]

    print("Central Density:", rho_c / 10 ** 3, r"$\frac{g}{cm^3}$")
    print("Central Temperature:", T_c / 10 ** 6, "million K")
    print("Radius:", surface_r / R_sun, r"$R_{sun}$")
    print("Mass:", state_values[M_index, -1] / M_sun, r"$M_{sun}$")
    print("Luminosity:", surface_L / L_sun, r"$L_{sun}$")
    print("Surface Temperature:", surface_T, "K")

    surface_index = int(surface_index)
    r_graph_values = r_values[:surface_index] / surface_r

    plt.plot(r_graph_values, state_values[rho_index, :surface_index] / rho_c, label=r"$\rho$")
    plt.plot(r_graph_values, state_values[T_index, :surface_index] / T_c, label="T")
    plt.plot(r_graph_values, state_values[M_index, :surface_index] / surface_M, label="M")
    plt.plot(r_graph_values, state_values[L_index, :surface_index] / surface_L, label="L")
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
    plt.plot(r_graph_values, P_total_values / P_total_values[0], label=r"$P_{total}$")
    plt.plot(r_graph_values, P_degeneracy_values / P_total_values[0], label=r"$P_{deg}$")
    plt.plot(r_graph_values, P_gas_values / P_total_values[0], label=r"$P_{gas}$")
    plt.plot(r_graph_values, P_photon_values / P_total_values[0], label=r"$P_{photon}$")
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
    plt.plot(r_graph_values, np.log10(10 * kappa_total_values), label=r"$\kappa_{total}$")
    plt.plot(r_graph_values, np.log10(10 * kappa_es_values), label=r"$\kappa_{es}$")
    plt.plot(r_graph_values, np.log10(10 * kappa_ff_values), label=r"$\kappa_{ff}$")
    plt.plot(r_graph_values, np.log10(10 * kappa_H_minus_values), label=r"$\kappa_{H-}$")
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
    plt.plot(r_graph_values, L_total_prime / surface_L * surface_r, label=r"$\frac{L}{dr}$")
    plt.plot(r_graph_values, L_proton_proton_prime_values / surface_L * surface_r,
             label=r"$\frac{L_{proton-proton}}{dr}$")
    plt.plot(r_graph_values, L_CNO_prime_values / surface_L * surface_r, label=r"$\frac{L_{CNO}}{dr}$")
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
    plt.plot(r_graph_values[:-1], dlogP_dlogT_values)
    plt.xlim(0, 1)
    plt.xlabel(r"$r/R_{star}$")
    plt.ylabel(r"$\frac{dlogP}{dlogT}$")
    plt.title(r"$\frac{dlogP}{dlogT}$ Versus Radius Fraction")
    plt.savefig("../Graphs/" + name + "_dlogP_dlogT.png")
    plt.clf()
