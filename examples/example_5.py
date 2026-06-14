# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:02:04 2026

@author: Admin

Example 2: Middle Heater Case Study
Adapted to the updated Caloris object-oriented & decoupled solver framework.
"""

import numpy as np
import matplotlib.pyplot as plt

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.transient_solver import TransientSolver  # Decoupled implicit solver engine

# -----------------------------
# Material and geometry
# -----------------------------
rho = 2700        # kg/m³ for Al6061
r = 0.01          # rod radius [m]
A = np.pi * r**2  # cross-section [m²]

N_nodes = 7
node_labels = [f"N{i}" for i in range(N_nodes)]
L_links = [0.01, 0.02, 0.02, 0.02, 0.02, 0.01]  # 6 links

# Compute link masses
link_masses = [rho * A * L for L in L_links]

# Node positions for plotting
x_positions = [0]
for L in L_links:
    x_positions.append(x_positions[-1] + L)
x_positions = np.array(x_positions)

# -----------------------------
# Example 2: Middle Heater
# -----------------------------
nodes2 = []
for i in range(N_nodes):
    if i == 0 or i == N_nodes - 1:
        # REPLACED Thermostat: Using the unified Node class with a 'temperature' boundary condition
        nodes2.append(Node(
            label=node_labels[i], 
            initial_temperature=293.15, 
            boundary_type='temperature', 
            boundary_value=293.15, 
            specific_heat='Al6061'
        ))
    elif i == N_nodes // 2:
        # REPLACED Heater: Using the unified Node class with a constant 'heat_input' boundary condition
        nodes2.append(Node(
            label=node_labels[i], 
            initial_temperature=293.15, 
            boundary_type='heat_input', 
            boundary_value=100.0,  # 100 W constant heat generation
            specific_heat='Al6061'
        ))
    else:
        # FIXED signatures: temperature -> initial_temperature, material_specific_heat -> specific_heat
        nodes2.append(Node(
            label=node_labels[i], 
            initial_temperature=293.15, 
            specific_heat='Al6061'
        ))

# Assign node masses
for i in range(N_nodes):
    m_node = 0.0
    if i > 0:
        m_node += 0.5 * link_masses[i-1]
    if i < N_nodes - 1:
        m_node += 0.5 * link_masses[i]
    nodes2[i].mass = m_node

# Connections
connections2 = []
for i in range(N_nodes - 1):
    connections2.append(Connection(
        node_i=nodes2[i],
        node_j=nodes2[i+1],
        connection_type='conduction',  # FIXED: type_ -> connection_type
        L=L_links[i],
        A=A,
        material_conductivity='Al6061'
    ))

# Network Graph
net2 = Network(nodes2, connections2, spread=1.0)
t_max = 150.0
dt = 1.5

# REPLACED legacy solver loop: Invoking the implicit TransientSolver object
solver2 = TransientSolver(net2)
result2 = solver2.solve(t_max=t_max, dt=dt, verbose=False)

# Extract structured outcome arrays from TransientResult
time_points2 = result2.time
T_history2 = result2.T_history  # Array layout shape: (n_steps, N_nodes)

# -----------------------------
# Plots Example 2
# -----------------------------
# 1. Temperature vs Time
fig1, ax1 = plt.subplots(figsize=(8, 5))
for i, label in enumerate(node_labels):
    # Mapping slice along axis-1 since layout is now (time, node)
    ax1.plot(time_points2, T_history2[:, i] - 273.15, label=label)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Temperature [°C]")
ax1.set_title("Example 2: T vs Time (middle heater)")
ax1.legend()
ax1.grid(True)
plt.savefig("example_2_time_evolution.png", bbox_inches='tight')

# 2. Temperature vs Position at selected times
time_slices2 = [0, 5, 10, 20, 150]
fig2, ax2 = plt.subplots(figsize=(8, 5))
for t_slice in time_slices2:
    idx = np.argmin(np.abs(time_points2 - t_slice))
    # Mapping horizontal slice across all nodes at time step `idx`
    ax2.plot(x_positions, T_history2[idx, :] - 273.15, label=f"t={time_points2[idx]:.0f}s", marker='o')
ax2.set_xlabel("Position [m]")
ax2.set_ylabel("Temperature [°C]")
ax2.set_title("Example 2: T vs Position at different times")
ax2.legend()
ax2.grid(True)
plt.savefig("example_2_spatial_profiles.png", bbox_inches='tight')

net2.plot(save_path="my_network_layout.png")