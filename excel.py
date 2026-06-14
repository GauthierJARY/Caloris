# -*- coding: utf-8 -*-
"""
Excel configuration reader and CSV result exporter for Caloris
Created on Mon Apr  6 21:10:00 2026
@author: G.J
"""
import pandas as pd

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.steady_solver import SteadySolver


def load_network(excel_file):

    df_nodes = pd.read_excel(
        excel_file,
        sheet_name="nodes",
        engine="openpyxl"
    )

    df_links = pd.read_excel(
        excel_file,
        sheet_name="links",
        engine="openpyxl"
    )

    nodes = {}
    node_list = []

    # -----------------
    # NODES
    # -----------------

    for _, row in df_nodes.iterrows():

        if pd.isna(row["index"]):
            continue

        node = Node(
            label=row["label"],
            initial_temperature=row.get("initial_temperature", 300.0),
            constant_heat_input=row.get("constant_heat_input", 0.0),
            specific_heat=row.get("material_specific_heat"),
            mass=row.get("mass", 1.0),
            boundary_type=row.get("boundary_type"),
            boundary_value=row.get("boundary_value")
        )

        idx = int(row["index"])

        nodes[idx] = node
        node_list.append(node)

    # -----------------
    # CONNECTIONS
    # -----------------

    connection_list = []

    for _, row in df_links.iterrows():

        if pd.isna(row["Node_i index"]):
            continue

        kwargs = {}

        for field in [
            "L",
            "A",
            "h_c",
            "e_i",
            "e_j",
            "S_i",
            "S_j",
            "F_ij",
            "material_conductivity"
        ]:

            if field in row and pd.notna(row[field]):
                kwargs[field] = row[field]

        conn = Connection(
            node_i=nodes[int(row["Node_i index"])],
            node_j=nodes[int(row["Node_j index"])],
            connection_type=row["type"],
            **kwargs
        )

        connection_list.append(conn)

    return Network(
        nodes=node_list,
        connections=connection_list
    )


def export_results(network, result):

    nodes_df = pd.DataFrame({
        "Node": [n.label for n in network.nodes],
        "Temperature_K": result.T,
        "Temperature_C": result.T - 273.15
    })

    flux_rows = []

    for (i, j, typ), q in result.fluxes.items():

        flux_rows.append({
            "Node_i": i,
            "Node_j": j,
            "Type": typ,
            "Heat_Flux_W": q
        })

    flux_df = pd.DataFrame(flux_rows)

    nodes_df.to_csv("node_results.csv", index=False)
    flux_df.to_csv("flux_results.csv", index=False)
    
    
if __name__ == "__main__":

    excel_file = r"C:\Users\Admin\Desktop\Caloris\Caloris.xlsx"

    network = load_network(excel_file)

    solver = SteadySolver(network)

    result = solver.solve(verbose=False)

    export_results(network, result)

    print("Finished.")