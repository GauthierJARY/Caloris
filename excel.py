# -*- coding: utf-8 -*-
"""
Excel interface for Caloris thermal solver
Created on Mon Apr  6 18:55:29 2026
@author: G.J
"""

import pandas as pd

from Caloris.nodes import Node, Thermostat, Heater
from Caloris.connections import Connection
from Caloris.network import Network
import ast


def _to_float(val):
    """Convert spreadsheet values to float, even if it's a simple Excel formula."""

    if pd.isna(val):
        return 0.0
    if isinstance(val, str) and val.startswith("="):
        # remove '=' and safely evaluate simple math expressions
        expr = val[1:]
        try:
            return float(ast.literal_eval(expr.replace("^", "**")))
        except Exception:
            raise ValueError(f"Cannot evaluate formula: {val}")
    return float(val)


def load_nodes(file):
    """Read nodes sheet and construct Node objects, robust to empty cells."""
    df = pd.read_excel(file, sheet_name="nodes", header=0)

    nodes = []

    for _, row in df.iterrows():
        if pd.isna(row["class"]):
            continue  # skip completely empty rows

        label = str(row["label"]).strip()
        node_class = str(row["class"]).strip()

        if node_class == "Node":
            node = Node(label=label)

        elif node_class == "Thermostat":
            T = _to_float(row["temperature"])
            node = Thermostat(label=label, fixed_temperature=T)

        elif node_class == "Heater":
            Q = _to_float(row["behaviour"])  # now returns 0 if blank
            node = Heater(label=label, behaviour_func=Q)

        else:
            raise ValueError(f"Unknown node class: {node_class}")

        nodes.append(node)

    return nodes


def load_connections(file, nodes):
    """Read links sheet and construct Connection objects."""

    df = pd.read_excel(file, sheet_name="links", header=0)

    connections = []

    for _, row in df.iterrows():

        if pd.isna(row["Node_i index"]):
            continue

        i = int(row["Node_i index"])
        j = int(row["Node_j index"])

        type_ = str(row["type"]).strip()

        kwargs = {}

        L = _to_float(row["L"])
        A = _to_float(row["A"])
        h_c = _to_float(row["h_c"])
        material_conductivity = row["material_conductivity"]
        e_i = _to_float(row["e_i"])
        e_j = _to_float(row["e_j"])
        S_i = _to_float(row["S_i"])
        S_j = _to_float(row["S_j"])
        F_ij = _to_float(row["F_ij"])
        if L is not None:
            kwargs["L"] = L

        if A is not None:
            kwargs["A"] = A

        if h_c is not None:
            kwargs["h_c"] = h_c
            
        if e_i is not None:
            kwargs["e_i"] = e_i
            
        if e_j is not None:
            kwargs["e_j"] = e_j
            
        if S_i is not None:
            kwargs["S_i"] = S_i
            
        if S_j is not None:
            kwargs["S_j"] = S_j
 
        if F_ij is not None:
            kwargs["F_ij"] = F_ij
            
        # if not pd.isna(material_conductivity):
        #     kwargs["material_conductivity"] = str(material_conductivity)
        if not pd.isna(material_conductivity):
            if isinstance(material_conductivity, (int, float)):
                kwargs["material_conductivity"] = float(material_conductivity)
            else:
                # Could still be a numeric string like "14.2"
                try:
                    kwargs["material_conductivity"] = float(material_conductivity)
                except (ValueError, TypeError):
                    kwargs["material_conductivity"] = str(material_conductivity).strip()
            
        connection = Connection(
            nodes[i],
            nodes[j],
            type_=type_,
            **kwargs
        )

        connections.append(connection)

    return connections


def load_network(file):
    """Construct full thermal network from Excel file."""

    nodes = load_nodes(file)
    connections = load_connections(file, nodes)

    return Network(nodes, connections)


def save_results_to_excel(file, T, fluxes, node_to_idx, G_matrix, convergence):
    """
    Write results into an Excel sheet in a clean tabular format.
    Works with Network.solve_steady outputs.
    """

    # Reverse mapping (idx → node label)
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # -------------------------
    # 1. Node temperatures
    # -------------------------
    df_T = pd.DataFrame({
        "Node": [idx_to_node[i] for i in range(len(T))],
        "Temperature [K]": T
    })

    # -------------------------
    # 2. Fluxes
    # -------------------------
    flux_rows = []
    for (i, j), val in fluxes.items():
        flux_rows.append({
            "From": i,
            "To": j,
            "Heat Flux [W]": val
        })
    df_flux = pd.DataFrame(flux_rows)

    # -------------------------
    # 3. G matrix
    # -------------------------
    df_G = pd.DataFrame(G_matrix)
    df_G.index = [idx_to_node[i] for i in range(len(G_matrix))]
    df_G.columns = [idx_to_node[i] for i in range(len(G_matrix))]

    # -------------------------
    # 4. Convergence
    # -------------------------
    # If convergence is a list of errors, report max error and iterations
    if isinstance(convergence, list):
        df_conv = pd.DataFrame({
            "Metric": ["Max error", "Iterations"],
            "Value": [max(convergence), len(convergence)]
        })
    else:  # dict-like
        df_conv = pd.DataFrame({
            "Metric": ["Converged", "Iterations"],
            "Value": [
                convergence.get("converged", None),
                convergence.get("iterations", None)
            ]
        })

    # -------------------------
    # Write to Excel
    # -------------------------
    with pd.ExcelWriter(file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_T.to_excel(writer, sheet_name="results", startrow=0, index=False)
        df_flux.to_excel(writer, sheet_name="results", startrow=0, startcol=4, index=False)
        df_G.to_excel(writer, sheet_name="results", startrow=10)
        df_conv.to_excel(writer, sheet_name="results", startrow=10, startcol=6, index=False)
        
if __name__ == "__main__":

    net = load_network("Caloris.xlsx")
    res = net.solve_steady(verbose=False)
    T = res["T"]
    fluxes = res["fluxes"]
    convergence_history = res["convergence"]
    # save_results_to_excel(
    # "Caloris_results.xlsx",
    # T,
    # fluxes,
    # net.node_to_idx,
    # net._last_G,
    # {"converged": True, "iterations": len(res["convergence"])}
    # )
    print((T-273))

