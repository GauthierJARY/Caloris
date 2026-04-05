# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.
"""

import numpy as np
from scipy.interpolate import interp1d
# import networkx as nx
import matplotlib.pyplot as plt


#####################################
# --- Material DATA from NIST ---   #
#####################################

# --- 6061-T6 Aluminum ---
def lambda_aluminium_6061(T):
    logT = np.log10(T)
    a, b, c, d, e, f, g, h, i = 0.07918, 1.0957, -0.07277, 0.08084, 0.02803, -0.09464, 0.04179, -0.00571, 0.0
    poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
    return 10**poly
# --- 6063-T5 Aluminum ---
def lambda_aluminium_6063(T):
    logT = np.log10(T)
    a, b, c, d, e, f, g, h, i = 22.401433, -141.13433, 394.95461, -601.15377, 547.83202, -305.99691, 102.38656, -18.810237, 1.4576882
    poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
    return 10**poly
    # --- Fonction lambda SST304 ---
def lambda_SST304(T):
        logT = np.log10(T)
        a,b,c,d,e,f,g,h,i = -1.4087,1.3982,0.2543,-0.6260,0.2334,0.4256,-0.4658,0.1650,-0.0199
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    # --- Fonction lambda SST304L ---
def lambda_SST304L(T):
        logT = np.log10(T)
        a,b,c,d,e,f,g,h,i = -1.4087,1.3982,0.2543,-0.6260,0.2334,0.4256,-0.4658,0.1650,-0.0199
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    # --- Fonction lambda GFRP ---
def lambda_CFRP_warp(T):
        logT = np.log10(T)
        a,b,c,d,e,f,g,h,i = -2.64827, 8.80228, -24.8998, 41.1625, -39.8754, 23.1778, -7.95635, 1.48806, -0.11701
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    # --- Fonction lambda Cu RRR=50 ---
def lambda_Cu_RRR50(T):
        logT = np.log10(T)
        a,b,c,d,e,f,g,h,i = 1.8743,-0.41538,-0.6018,0.13294, 0.26426, -0.0219, -0.051276, 0.0014871, 0.003723
        poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
        return 10**poly
def lambda_Cu_RRR10(T):
    return lambda_Cu_RRR50(T)/5
def lambda_Cu_RRR20(T):
    return lambda_Cu_RRR50(T)/2.5
def lambda_PEEK(T):
    # Coefficients a1 à a6
    coefficients = [1.0636976e-1, -1.6340006e-1, 9.4941322e-2, -2.4117988e-2, 2.8797748e-3, -1.3025208e-4 ]
    ln_term = np.log(T + 1)  # ln(T + 1)
    # Calcule A(T) = sum(ai * [ln(T+1)]^i) pour i de 1 à 6
    A_T = sum(coeff * ln_term**(i+1) for i, coeff in enumerate(coefficients))
    A_T = np.maximum(A_T, 0.0108)
    return A_T
    
    # --- Lambda en fonction du matériau de la liaison ---
def lambda_material_dispatch(T, material):
        if material == 'Al6061':
            return lambda_aluminium_6061(T)
        elif material == 'Al6063':
            return lambda_aluminium_6063(T)
        elif material == 'sst304':
            return lambda_SST304(T)
        elif material == 'cfrp':
            return lambda_CFRP_warp(T)
        elif material == 'sst304L':
            return lambda_SST304L(T)
        elif material == 'Cu_RRR50':
            return lambda_Cu_RRR50(T)
        elif material == 'Cu_RRR10':
            return lambda_Cu_RRR10(T)
        elif material == 'Cu_RRR20':
            return lambda_Cu_RRR20(T)
        elif material == 'PEEK':
            return lambda_PEEK(T)
        else:
            return 15.0  # valeur par défaut
        
        
def custom_conductance_200kN(T):
    """A sample function for direct_G that depends on temperature."""
    # return 0.1 + 0.005 * T
    return 2000
        
# --- Conductance function dispatcher ---

def contact_conductance_dispatch(T,name):
    if name == 'custom_conductance_200kN':
        return custom_conductance_200kN(T)
    elif name == 'custom_conductance_100kN':
        return custom_conductance_100kN(T)
    else:
        raise ValueError(f"❌ Unknown conductance function: '{name}'")
        
#*******************************#
# Comstetic functions Visuals   #
#*******************************#

def plot_network(connections, T_final, fluxes, node_to_idx, thermostat_node, Q, length_support, material_support_or_title, area_support):
   
    G = nx.DiGraph()
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    for node in node_to_idx: 
        G.add_node(node)
    for conn in connections:
        i, j, mode = conn[:3]
        G.add_edge(i, j, mode=mode)
    
    boundary_node = "Boundary"
    G.add_node(boundary_node)
    
    for idx, q in enumerate(Q):
        node = idx_to_node[idx]
        if q != 0 and node != thermostat_node:
            G.add_edge(boundary_node, node, flux=q, mode="source")

    for (i, j), flux in fluxes.items():
        if G.has_edge(i, j):
            G.edges[i, j]['flux'] = flux
        else:
            G.add_edge(i, j, flux=flux, mode="calculated")

    # --- Positions manuelles ---
    pos_manual = {
        10: (-0.5, 6),
        9: (-0.3, 5),
        8: (0, 5.2),
        7: (0, 4),
        5: (-0.2, 3),
        6: (0.1, 3),
        4: (-0.5, 3.5),
        14: (0.5, 2.3),
        3: (-0.5, 2),
        2: (-0.5, 0.3),
        11: (0.5, 0.3),
        13: (-0.3, 0),
        12: (0.3, 0),
        1: (0, 0),
        boundary_node: (0.3, 5)
    }

    # Étirement vertical
    stretch_y = 0.5
    for node in pos_manual:
        x, y = pos_manual[node]
        pos_manual[node] = (x, y * stretch_y)

    # --- Génération des positions automatiques autour des positions fixes ---
    all_nodes = list(G.nodes())
    fixed_nodes = list(pos_manual.keys())
    pos_init = pos_manual.copy()
    
    # Calcul layout complet avec nœuds fixes
    pos_complete = nx.spring_layout(G, pos=pos_init, fixed=fixed_nodes, seed=42, k=0.5)
    
    # pos_complete contient les positions finales (fixes + auto)
    pos = pos_complete


    # --- Affichage ---
    plt.figure(figsize=(7, 5))

    # Arêtes avec flux uniquement
    edges_with_flux = [(i, j) for (i, j) in G.edges() if 'flux' in G.edges[i, j]]
    nx.draw_networkx_edges(G, pos, edgelist=edges_with_flux, arrowstyle='-|>', arrowsize=12)

    # Couleur des nœuds
    node_colors = []
    for n in G.nodes():
        if n == boundary_node or n == thermostat_node:
            node_colors.append('dimgray')
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='k')

    # Labels des nœuds
    labels = {}
    for n in G.nodes():
        if n == boundary_node:
            labels[n] = 'Boundary'
        else:
            idx = node_to_idx.get(n, None)
            if idx is not None:
                labels[n] = f"N°{n}\n{T_final[idx]:.1f} K"
            else:
                labels[n] = f"N°{n}"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    # Labels des fluxs
    edge_labels = {(i, j): f"{G.edges[i, j]['flux']*1000:.1f} mW" for (i, j) in edges_with_flux}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    if length_support != 0:
        plt.title(f"Thermal network with {sum(Q)*1000:.1f} mW dissipated\n"
                  f"{length_support*100} cm support | {material_support_or_title} | {area_support*10000} cm² support")
    else: 
        plt.title(f"Thermal network with {sum(Q)*1000:.1f} mW dissipated\n"
                  f"{material_support_or_title}")
    # plt.title(f"Thermal network with {sum(Q)*1000:.1f} mW dissipated\n"
    #           f"With CEA copper braiding on JT I/F")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    
def plotting_lambda_curves(visu):
    if visu: 
        # Create a list of all the functions and their labels for easy plotting
        functions_to_plot = [
            (lambda_aluminium_6061, 'Aluminium 6061-T6'),
            (lambda_aluminium_6063, 'Aluminium 6063-T5'),
            (lambda_SST304, 'SST 304'),
            (lambda_SST304L, 'SST 304L'),
            (lambda_CFRP_warp, 'CFRP (warp)'),
            (lambda_Cu_RRR50, 'Copper RRR=50'),
            (lambda_Cu_RRR10, 'Copper RRR=10'),
            (lambda_PEEK, 'PEEK'),
            (lambda_Cu_RRR20, 'Copper RRR=20')
        ]
        # Create a range of T (temperature) values to plot
        # Using np.logspace for a logarithmic scale on the x-axis to better visualize the data,
        # as the functions are defined with log10(T).
        T_values = np.logspace(np.log10(1), np.log10(300), 1000)
        # Create the plot
        plt.figure(figsize=(10, 7))
        # Loop through the list and plot each function
        for func, label in functions_to_plot:
            plt.plot(T_values, func(T_values), label=label)
        # Add plot labels and title
        plt.title('Thermal Conductivity of Various Materials')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Thermal Conductivity ($\lambda$)')
        plt.xscale('log') # Use a logarithmic scale for the x-axis
        plt.yscale('log') # Use a logarithmic scale for the y-axis
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.ylim(0.05, 10**3)
        plt.xlim(4, 300)
        plt.show()



# results_writer.py

sigma = 5.670374419e-8  # W/m²K⁴

def save_thermal_results(filename, T, fluxes, node_to_idx, connections, spread=1.0):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Thermal Network Results\n")
        f.write("="*75 + "\n\n")

        for node_i, idx_i in node_to_idx.items():
            f.write(f"Node {node_i} (Temperature: {T[idx_i]:.2f} K)\n")
            f.write("-"*95 + "\n")
            f.write(f"| {'To Node':<8} | {'G [W/K]':<11} | {'λ [W/m·K]':<12} | {'Heat Flux [W]':<16} | {'Type':<9} | {'Material':<10} |\n")
            f.write(f"|{'-'*10}|{'-'*13}|{'-'*14}|{'-'*18}|{'-'*11}|{'-'*12}|\n")

            for connection in connections:
                i, j = connection[0], connection[1]
                conn_type = connection[2]

                if i != node_i and j != node_i:
                    continue

                idx_j = node_to_idx[j] if i == node_i else node_to_idx[i]
                node_j = j if i == node_i else i

                if (node_i, node_j) in fluxes:
                    flux_val = -fluxes[(node_i, node_j)]  # EU convention negative = give heat
                    arrow = "→"
                elif (node_j, node_i) in fluxes:
                    flux_val = fluxes[(node_j, node_i)]  # positive if receive heat
                    arrow = "←"
                else:
                    flux_val = 0.0
                    arrow = "→"

                # Default values
                G_ij = 0.0
                material_str = "--"
                lambda_val_str = "--"

                # Compute G and lambda if needed
                if conn_type == 'conduction':
                    L, A, material = connection[3], connection[4], connection[5]
                    T_avg = 0.5 * (T[idx_i] + T[idx_j])
                    lambda_val = lambda_material_dispatch(T_avg, material)
                    G_ij = lambda_val * spread * A / L
                    material_str = material
                    lambda_val_str = f"{lambda_val:.2e}"
                elif conn_type == 'contact':
                    A, h_c = connection[3], connection[4]
                    G_ij = A * h_c
                elif conn_type == 'direct_G':
                    G_ij = contact_conductance_dispatch(0.5 * (T[idx_i] + T[idx_j]), connection[3]) * (1 if spread >= 1 else 0.5)
                elif conn_type == 'radiation':
                    epsilon, A, F_ij = connection[3], connection[4], connection[5]
                    T_i, T_j = T[idx_i], T[idx_j]
                    G_ij = 4 * sigma * epsilon * A * F_ij * (T_i**3 + T_j**3) / 2

                f.write(f"| {arrow} {node_j:<6} | {G_ij:>11.4e} | {lambda_val_str:>12} | {flux_val:>+16.4e} | {conn_type:<9} | {material_str:<10} |\n")

            f.write("-"*95 + "\n\n")
            
def display_thermal_results(T_final, fluxes, node_to_idx, thermostat_node, Q, connections, length_support, material_support, area_support, convergence_history, lambda_history):
    
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Affichage des températures
    print("\n🌡️ Final Node Temperatures:")
    for idx, Tn in enumerate(T_final):
        node = idx_to_node[idx]
        is_heater = "🔥" if Q[idx] > 0 else ""
        is_thermo = "❄️" if node == thermostat_node else ""
        print(f"Node {node:2d} : {Tn:.2f} K {is_heater}{is_thermo}")
    
    # Affichage des flux
    print("\n🔀 Fluxes between nodes:")
    for (i, j), flux in fluxes.items():
        cryostat_flag = "⚗️" if i == thermostat_node else ""
        print(f"Flux {i} → {j} : {flux:.3f} W {cryostat_flag}")
    
    # Plot convergence and parameter history
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max error (K)', color=color)
    ax1.plot(convergence_history, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Thermal conductivity (W/m/K)', color=color)
    ax2.plot(lambda_history, 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Convergence history and parameter evolution')
    fig.tight_layout()
    plt.show()
    
def get_net_flux_to_node(node_label, fluxes): 
    """
    BUG : node label required and node node idx !
    Calcule le flux net entrant dans un nœud donné.

    Args:
        node_label (int or str): Identifiant du nœud (numérique ou textuel selon ton graphe).
        fluxes (dict): Dictionnaire des flux {(i, j): flux_pos}, où le flux va de i vers j.

    Returns:
        float: Flux net entrant (positif = entre, négatif = sort).
    """
    net_flux = 0.0
    for (i, j), flux in fluxes.items():
        if j == node_label:
            net_flux += flux  # Flux entrant
        elif i == node_label:
            net_flux -= flux  # Flux sortant
    return net_flux

def get_total_flux_in_to_node(node_label, fluxes):
    """
    Calcule le flux total entrant dans un nœud donné.

    Args:
        node_label (int or str): Identifiant du nœud (numérique ou textuel selon ton graphe).
        fluxes (dict): Dictionnaire des flux {(i, j): flux_pos}, où le flux va de i vers j.

    Returns:
        float: Flux total entrant (positif).
    """
    total_flux_in = 0.0
    for (i, j), flux in fluxes.items():
        if j == node_label:
            total_flux_in += flux  # Flux entrant
    return total_flux_in

