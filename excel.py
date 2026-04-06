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


def _to_float(val):
    """Safely convert spreadsheet values to float."""
    if pd.isna(val):
        return None
    return float(val)


def load_nodes(file):
    """Read nodes sheet and construct Node objects."""

    df = pd.read_excel(file, sheet_name="nodes", header=4)

    nodes = []

    for _, row in df.iterrows():

        if pd.isna(row["class"]):
            continue

        label = str(row["label"])
        node_class = str(row["class"]).strip()

        if node_class == "Node":

            node = Node(label=label)

        elif node_class == "Thermostat":

            T = _to_float(row["temperature"])

            node = Thermostat(
                label=label,
                fixed_temperature=T
            )

        elif node_class == "Heater":

            Q = _to_float(row["behaviour"])

            node = Heater(
                label=label,
                behaviour_func=Q
            )

        else:
            raise ValueError(f"Unknown node class: {node_class}")

        nodes.append(node)

    return nodes


def load_connections(file, nodes):
    """Read links sheet and construct Connection objects."""

    df = pd.read_excel(file, sheet_name="links", header=4)

    connections = []

    for _, row in df.iterrows():

        if pd.isna(row["Node_i index"]):
            continue

        i = int(row["Node_i index"])
        j = int(row["Node_j"])

        type_ = str(row["type"]).strip()

        kwargs = {}

        L = _to_float(row["L"])
        A = _to_float(row["A"])
        h_c = _to_float(row["h_c"])
        material = row["material"]

        if L is not None:
            kwargs["L"] = L

        if A is not None:
            kwargs["A"] = A

        if h_c is not None:
            kwargs["h_c"] = h_c

        if not pd.isna(material):
            kwargs["material"] = str(material)

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


if __name__ == "__main__":

    net = load_network("Caloris.xlsx")

    T, fluxes, convergence = net.solve_steady(verbose=True)

    print("Temperatures:")
    print(T-273)
