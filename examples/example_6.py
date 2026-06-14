# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:06:35 2026

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:48:15 2026

Example 3: Radiation Heat Transfer Case Studies
Adapted to the updated Caloris object-oriented & decoupled solver framework.
"""

import numpy as np

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.steady_solver import SteadySolver  # Imported decoupled steady solver engine

print('\nRadiative examples')

# =============================================================================
# Sub-Example 3.1: Only two plates facing each other
# =============================================================================
nodes31 = [
    # REPLACED Thermostat with the unified Node class
    Node(label='VG1', initial_temperature=300.0, boundary_type='temperature', boundary_value=300.0),
    Node(label='space', initial_temperature=50.0, boundary_type='temperature', boundary_value=50.0)
]

connections31 = [
    # FIXED: type_ -> connection_type
    Connection(nodes31[0], nodes31[1], connection_type='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
]

net31 = Network(nodes31, connections31)

# REPLACED legacy solver call with the standalone SteadySolver engine
solver31 = SteadySolver(net31)
res31 = solver31.solve(verbose=False)

# Support either object-attribute or dictionary access depending on your SteadySolver output
T31 = res31["T"] if isinstance(res31, dict) else res31.T
fluxes31 = res31["fluxes"] if isinstance(res31, dict) else res31.fluxes

print('Cas3.1')
# FIXED: Flux dictionary keys are now 3-tuples including the connection type
flux_key31 = (nodes31[0].label, nodes31[1].label, 'radiation')
print(f"{int(T31[0])}K -> {int(fluxes31[flux_key31])}W -> {int(T31[1])}K")


# =============================================================================
# Sub-Example 3.2: Two plates facing each other with a floating shield in between
# =============================================================================
nodes32 = [
    Node(label='VG1', initial_temperature=300.0, boundary_type='temperature', boundary_value=300.0),
    Node(label='VG2', initial_temperature=293.15),  # Unbounded floating node
    Node(label='space', initial_temperature=50.0, boundary_type='temperature', boundary_value=50.0)
]

connections32 = [
    Connection(nodes32[0], nodes32[1], connection_type='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
    Connection(nodes32[1], nodes32[2], connection_type='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
]

net32 = Network(nodes32, connections32)

# FIXED: temperature -> initial_temperature
T_nodes32 = np.array([node.initial_temperature for node in nodes32])

# FIXED: compute_fluxes signature only accepts T_nodes (G matrix parameter removed)
G32 = net32.build_G(T_nodes32)
fluxes32_initial = net32.compute_fluxes(T_nodes32)

# Solve steady-state
solver32 = SteadySolver(net32)
res32 = solver32.solve(verbose=False)

T32 = res32["T"] if isinstance(res32, dict) else res32.T
fluxes32 = res32["fluxes"] if isinstance(res32, dict) else res32.fluxes

print('Cas3.2')
flux_key32_1 = (nodes32[0].label, nodes32[1].label, 'radiation')
flux_key32_2 = (nodes32[1].label, nodes32[2].label, 'radiation')
print(f"{int(T32[0])}K -> {int(fluxes32[flux_key32_1])}W -> "
      f"{int(T32[1])}K -> {int(fluxes32[flux_key32_2])}W -> {int(T32[2])}K")


# =============================================================================
# Sub-Example 3.4: Active heater plate radiating to space via shield
# =============================================================================
nodes34 = [
    # REPLACED Heater with unified Node class using heat_input boundary conditions
    Node(label='VG1', initial_temperature=293.15, boundary_type='heat_input', boundary_value=153.0),
    Node(label='VG2', initial_temperature=293.15),
    Node(label='space', initial_temperature=50.0, boundary_type='temperature', boundary_value=50.0)
]

connections34 = [
    Connection(nodes34[0], nodes34[1], connection_type='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
    Connection(nodes34[1], nodes34[2], connection_type='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
]

net34 = Network(nodes34, connections34)

solver34 = SteadySolver(net34)
res34 = solver34.solve(verbose=False)

T34 = res34["T"] if isinstance(res34, dict) else res34.T
fluxes34 = res34["fluxes"] if isinstance(res34, dict) else res34.fluxes

print('Cas3.4')
flux_key34_1 = (nodes34[0].label, nodes34[1].label, 'radiation')
flux_key34_2 = (nodes34[1].label, nodes34[2].label, 'radiation')
print(f"{int(T34[0])}K -> {int(fluxes34[flux_key34_1])}W -> {int(T34[1])}K -> {int(fluxes34[flux_key34_2])}W -> {int(T34[2])}K")