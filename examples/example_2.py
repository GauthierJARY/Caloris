# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:48:55 2026

@author: Admin
"""
# =============================================================
# Example 2: Heat transfer transient case study (Updated Workflow)
# =============================================================

import numpy as np
from Caloris.nodes import Node  
from Caloris.connections import Connection
from Caloris.network import Network

# 1. IMPORT THE NEW SOLVER
# (Adjust this import path if you saved TransientSolver in a different file)
from Caloris.transient_solver import TransientSolver 


# --- Define example behaviour functions ---
def heater_behaviour(T, Q_in):
    return 0.1  # constant power input (W)

def cryostat_behaviour(T, Q_in):
    return -Q_in  # simple passive sink


# --- Define nodes ---
node1 = Node(
    label='T1', 
    initial_temperature=8.0,
    boundary_type='temperature',
    boundary_value=4.0,
    specific_heat='Al6061',  
    mass=1.0                 
) 

node2 = Node(
    label='H1', 
    initial_temperature=40.0,
    boundary_type='heat_input',
    boundary_function=heater_behaviour,
    specific_heat='Al6061',  
    mass=1.0
)

node3 = Node(
    label='C1', 
    initial_temperature=150.0,
    boundary_type='heat_input',
    boundary_function=cryostat_behaviour,
    specific_heat='Al6061',  
    mass=1.0
)

nodes = [node1, node2, node3]


# --- Define connections ---
conn1 = Connection(
    node_i=node1, 
    node_j=node2, 
    connection_type='conduction',
    A=0.01, 
    L=1.0, 
    material_conductivity='Al6061'
)

conn2 = Connection(
    node_i=node2, 
    node_j=node3, 
    connection_type='conduction',
    A=0.01, 
    L=1.0, 
    material_conductivity='Al6061'
)

connections = [conn1, conn2]


# --- Build network graph ---
net = Network(nodes, connections)


# --- 2. SOLVE USING THE NEW SOLVER OBJECT ---
solver = TransientSolver(net)
result = solver.solve(t_max=3600.0, dt=500, verbose=False)

# 3. EXTRACT ARRAYS FROM THE TRANSIENT RESULT OBJECT
T_hist = result.T_history
time_points = result.time


# --- Print results ---
print("\nFinal Temperatures:")
for node, T in zip(nodes, T_hist[-1]):
    print(f"{node.label}: {T:.2f} K")

print("\nTemperature evolution:")
for t, T in zip(time_points, T_hist):
    T_str = " | ".join(f"{Ti:.2f}" for Ti in T)
    print(f"t={t:.2f} s: {T_str}")