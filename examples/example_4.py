# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:10:52 2026

@author: Admin
"""

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.solver import SteadySolver



# ==========================================================
# Example 4 : Steady conduction in a large plate
# from : https://drive.uqu.edu.sa/_/kmguedri/files/A-HT-1-Chap5.pdf
# ==========================================================

nodes = [

    Node(
        label="0",
        boundary_type="temperature",
        boundary_value=273.0
    ),

    Node(
        label="1",
        constant_heat_input=5e6 * 10 * 4e-2  # W
    ),

    Node(
        label="2"
    ),

    Node(
        label="inf",
        boundary_type="temperature",
        boundary_value=303.0
    )

]

connections = [

    Connection(
        nodes[0],
        nodes[1],
        connection_type="conduction",
        L=2e-2,
        A=10,
        material_conductivity=28
    ),

    Connection(
        nodes[1],
        nodes[2],
        connection_type="conduction",
        L=2e-2,
        A=10,
        material_conductivity="Uranium"
    ),

    Connection(
        nodes[2],
        nodes[3],
        connection_type="convection",
        h_c=45,
        A=10
    )

]

network = Network(nodes, connections)

solver = SteadySolver(network)

result = solver.solve(verbose=False)

T2 = result.T[2] - 273.0

print(
    f"Temperature on the external boundary with air is "
    f"{T2:.1f} °C, to compare to analytical 136.0 °C"
)

print("\nTemperatures")

for node, T in zip(nodes, result.T):

    print(f"{node.label:10s} : {T:.2f} K")