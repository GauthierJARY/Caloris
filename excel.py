# -*- coding: utf-8 -*-
"""
Excel configuration reader and CSV result exporter for Caloris
Created on Mon Apr  6 21:10:00 2026
@author: G.J
Caloris Package - Excel Runner & Configuration Manager
Reads configuration from Excel, runs steady-state simulation, 
and writes results back into the workbook.
"""

import os
import pandas as pd

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.steady_solver import SteadySolver


def load_network_from_excel(excel_file):
    """
    Reads nodes, connections, and network-level parameters from Excel sheets.
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel configuration file not found at: '{excel_file}'")

    # 1. Lecture des Nœuds
    df_nodes = pd.read_excel(excel_file, sheet_name="nodes", engine="openpyxl")
    
    nodes_dict = {}  # Pour relier facilement les connexions par la suite
    node_list = []

    for _, row in df_nodes.iterrows():
        # Ignorer les lignes vides
        if pd.isna(row.get("label")):
            continue

        node = Node(
            label=str(row["label"]),
            number=int(row["number"]) if pd.notna(row.get("number")) else None,
            group_number=int(row["group_number"]) if pd.notna(row.get("group_number")) else 0,
            node_type=str(row.get("node_type", "D")).strip().upper(),
            initial_temperature=float(row.get("initial_temperature", 300.0)),
            qdot=float(row.get("qdot", 0.0)),
            mass=float(row.get("mass", 1.0)),
            capacitance=row.get("capacitance", 1.0),
            boundary_type=str(row["boundary_type"]).strip() if pd.notna(row.get("boundary_type")) else None,
            boundary_value=float(row["boundary_value"]) if pd.notna(row.get("boundary_value")) else None
        )

        nodes_dict[node.label] = node
        node_list.append(node)

    # 2. Lecture des Connexions (Links)
    df_links = pd.read_excel(excel_file, sheet_name="links", engine="openpyxl")
    connection_list = []

    for _, row in df_links.iterrows():
        if pd.isna(row.get("label")) or pd.isna(row.get("node_i")) or pd.isna(row.get("node_j")):
            continue

        node_i_label = str(row["node_i"])
        node_j_label = str(row["node_j"])

        if node_i_label not in nodes_dict or node_j_label not in nodes_dict:
            raise KeyError(
                f"Excel Link Error: Connection '{row['label']}' references node '{node_i_label}' "
                f"or '{node_j_label}' which is not defined in the 'nodes' sheet."
            )

        # Récupération dynamique de tous les paramètres optionnels de la ligne
        kwargs = {}
        possible_fields = [
            "S", "L", "material", "F_ij", "e_i", "e_j", "S_i", "S_j",
            "A", "h_c", "G_value", "G_function_name"
        ]
        
        for field in possible_fields:
            if field in row and pd.notna(row[field]):
                # Conversion propre si c'est numérique ou string
                val = row[field]
                kwargs[field] = val

        conn = Connection(
            label=str(row["label"]),
            node_i=nodes_dict[node_i_label],
            node_j=nodes_dict[node_j_label],
            connection_type=str(row["type"]).strip().upper(),
            **kwargs
        )

        connection_list.append(conn)

    # 3. Lecture optionnelle des paramètres globaux (ex: spread)
    spread = 1.0
    try:
        df_solver = pd.read_excel(excel_file, sheet_name="solver_params", engine="openpyxl")
        # On cherche une ligne où le paramètre est 'spread'
        for _, row in df_solver.iterrows():
            if str(row.get("parameter")).strip().lower() == "spread":
                spread = float(row.get("value", 1.0))
    except Exception:
        # Si l'onglet 'solver_params' n'existe pas, on garde la valeur par défaut 1.0
        pass

    network = Network(nodes=node_list, connections=connection_list, spread=spread)
    return network


def load_solver_config(excel_file):
    """
    Reads solver execution settings from the 'solver_params' sheet if available.
    """
    config = {"tol": 1e-5, "max_iter": 100, "relaxation": 0.5, "verbose": True}
    
    try:
        df_solver = pd.read_excel(excel_file, sheet_name="solver_params", engine="openpyxl")
        for _, row in df_solver.iterrows():
            param = str(row.get("parameter")).strip().lower()
            val = row.get("value")
            if pd.notna(val):
                if param in ["tol"]:
                    config["tol"] = float(val)
                elif param in ["max_iter"]:
                    config["max_iter"] = int(val)
                elif param in ["relaxation"]:
                    config["relaxation"] = float(val)
                elif param in ["verbose"]:
                    config["verbose"] = bool(val)
    except Exception:
        pass

    return config


def write_results_to_excel(excel_file, network, result):
    """
    Writes simulation results back into dedicated sheets of the Excel workbook.
    """
    # 1. Tableau des températures par nœud
    nodes_data = []
    for node, T in zip(network.nodes, result.T):
        nodes_data.append({
            "Number": node.number,
            "Label": node.label,
            "Type": node.node_type,
            "Boundary": node.boundary_type if node.node_type == "B" else "None",
            "Temperature [K]": T,
            "Temperature [C]": T - 273.15
        })
    df_nodes_res = pd.DataFrame(nodes_data)

    # 2. Tableau des flux par liaison
    flux_data = []
    for key, q in result.fluxes.items():
        node_i, node_j, link_type = key
        flux_data.append({
            "From": node_i,
            "To": node_j,
            "Type": link_type,
            "Heat Flux [W]": q
        })
    df_flux_res = pd.DataFrame(flux_data)

    # 3. Historique de convergence
    df_conv = pd.DataFrame({
        "Iteration": list(range(1, len(result.convergence) + 1)),
        "Max Error [K]": result.convergence
    })

    # Écriture dans le fichier Excel existant en conservant les onglets de configuration
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_nodes_res.to_excel(writer, sheet_name="Results_Temperatures", index=False)
        df_flux_res.to_excel(writer, sheet_name="Results_Fluxes", index=False)
        df_conv.to_excel(writer, sheet_name="Results_Convergence", index=False)

    print(f"Simulation results successfully written back to: {excel_file}")


if __name__ == "__main__":
    # Chemin vers ton fichier Excel principal
    excel_path = r"C:\Users\Admin\Desktop\Caloris\Caloris.xlsx"

    print("--- 1. Chargement du réseau depuis Excel ---")
    network, excel_path = load_network_from_excel(excel_path)
    print(f"Réseau chargé : {network.N} nœuds, {len(network.connections)} connexions.")

    print("--- 2. Chargement des paramètres du solveur ---")
    solver_config = load_solver_config(excel_path)

    print("--- 3. Exécution de la simulation stationnaire ---")
    solver = SteadySolver(network=network)
    result = solver.solve(**solver_config)

    print("--- 4. Exportation des résultats dans Excel ---")
    write_results_to_excel(excel_path, network, result)

    print("Process finished successfully.")
