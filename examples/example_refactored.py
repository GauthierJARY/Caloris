# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:45:39 2026

@author: Admin
"""

import numpy as np

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.solver import SteadySolver


nodes = []
nodes.append(

    Node(
        label="N0",
        initial_temperature=300,
        boundary_type="temperature",
        boundary_value=300
    )
)
nodes.append(

    Node(
        label=f"N{n_points}",
        initial_temperature=300,
        boundary_type="heat_input",
        boundary_value=1.0
    )
)
connections = []
for i in range(n_points):

    connections.append(

        Connection(
            nodes[i],
            nodes[i+1],
            connection_type="conduction",
            L=dx,
            A=area,
            material_conductivity="Cu_RRR50"
        )
    )
network = Network(
    nodes,
    connections
)
solver = SteadySolver(network)
res = solver.solve()
res.check_energy_balance(network)

