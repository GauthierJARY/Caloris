# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:03:18 2025

@author: jarygau

Example of use of the Lumped Parameter solver coded. We use the Finite Difference Method in 1D for steady state description of heat systems.

The analysis present two examples of use, compared with the analytical answer. 

Example 1 is the heat transfert across a copper tube, thermalized at one side.
Example 2 is the heat equilibrium between 2 radiating plates, ecranting shield. 
"""

import numpy as np
import matplotlib.pyplot as plt
from Caloris.nodes import Node, Thermostat, Heater
from Caloris.connections import Connection
from Caloris.network import Network
from scipy.optimize import fsolve
from Caloris.materials import lambda_material_dispatch

# =============================================
# Example 1: Heat transfer across a copper tube
# =============================================

# Paramètres du tube de cuivre
length = 1  # m
diameter = 1e-2  # m
area = np.pi * (diameter/2)**2
material = 'Al6061'
T_i = 300
k_cu = lambda_material_dispatch(T_i, material)  # Conductivité thermique du cuivre (W/m/K)
Q_total = 1  # Puissance dissipée (W)

# Nombre de points de discrétisation
n_points = 5

# Création des nœuds
nodes = []
nodes.append(Thermostat('0', fixed_temperature = T_i))
for i in range(1,n_points):
    x = i * length / n_points
    nodes.append(Node(str(i+1), temperature=4.2 if i == 0 else 300.0))
nodes.append(Heater(str(n_points+2), behaviour_func = Q_total))

# Création des connexions
connections = []
for i in range(n_points):
    dx = length / n_points
    connections.append(Connection(nodes[i], nodes[i+1], 'conduction',
                                L=dx, A=area, material=material))

# Création du réseau
network = Network(nodes, connections)

# Résolution
T, fluxes, _ = network.solve_steady()

# Solution analytique
x_analytical = np.linspace(0, length, 100)
T_analytical = T_i + (Q_total * x_analytical) / (k_cu * area)

# Tracé des résultats
plt.figure(figsize=(10, 6))

# Points numériques
x_numeric = np.array([i * length / n_points for i in range(n_points+1)])
plt.scatter(x_numeric, T, color='red', s=100, label='Solution numérique')

# Solution analytique
plt.plot(x_analytical, T_analytical, 'b-', linewidth=2, label='Solution analytique')

# Configuration du graphique
plt.title('Profil de température le long du tube de cuivre', fontsize=14)
plt.xlabel('Position le long du tube (m)', fontsize=12)
plt.ylabel('Température (K)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xticks(np.linspace(0, length, n_points+1))
# plt.yticks(np.linspace(T[0]-5, T[-1]+5, 5))

# Ajout des valeurs numériques sur le graphique
for i, (x, t) in enumerate(zip(x_numeric, T)):
    plt.text(x, t, f'{t:.1f} K', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Affichage des résultats
print("Résultats numériques:")
for i, (x, t) in enumerate(zip(x_numeric, T)):
    print(f"Position {x:.2f} m: {t:.2f} K")

print(f"\nSolution analytique à x={length:.2f} m: {T_analytical[-1]:.2f} K")
print(f"Solution numérique à x={length:.2f} m: {T[-1]:.2f} K")

# =============================================
# Example 2: Heat transfer transient case study
# =============================================

import numpy as np
from Caloris.nodes import Thermostat, Node, Cryostat, Heater
from Caloris.connections import Connection
from Caloris.network import Network


# --- Define example behaviour functions ---
def heater_behaviour(T, Q_in):
    return 0.1  # constant power input (W)

def cryostat_behaviour(T, Q_in):
    # Removes as much heat as received, tries to stabilize at 0°C
    return -Q_in  # simple passive sink

# --- Define nodes ---
node1 = Thermostat(label='T1', fixed_temperature=4.0, material='Al6061') 
node2 = Heater(label='H1', material='Al6061', behaviour_func=heater_behaviour)
node3 = Node(label='C1', material='Al6061')

nodes = [node1, node2, node3]

# --- Define connections (unit length & area assumed) ---
conn1 = Connection(node_i=node1, node_j=node2, type_='conduction', A=0.01, L=1, material = 'Al6061')
conn2 = Connection(node_i=node2, node_j=node3, type_='conduction', A=0.01, L=1, material = 'Al6061')

connections = [conn1, conn2]

# --- Build network ---
net = Network(nodes, connections)

# --- Solve transient ---
T_hist, time_points = net.solve_transient(t_max=3600.0, dt=500, verbose=False)

# --- Print results ---
print("\nFinal Temperatures:")
for node, T in zip(nodes, T_hist[-1]):
    print(f"{node.label}: {T:.2f} K")

print("\nTemperature evolution:")
for t, T in zip(time_points, T_hist):
    T_str = " | ".join(f"{Ti:.2f}" for Ti in T)
    print(f"t={t:.2f} s: {T_str}")


# -*- coding: utf-8 -*-
"""
Transient thermal network: 7-node aluminum rod
- Example 1: Step change at boundaries
- Example 2: Heater in middle
- Plots: T vs time and T vs position (°C)
"""

import numpy as np
import matplotlib.pyplot as plt
from Caloris.nodes import Node, Thermostat, Heater
from Caloris.network import Network
from Caloris.connections import Connection

# -----------------------------
# Material and geometry
# -----------------------------
rho = 2700        # kg/m³ for Al6061
r = 0.01        # rod radius [m]
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
# Example 1: Step change at boundaries
# -----------------------------
T0 = 273.15+100  # Initial temperature [K]
T1 = 273.15+200  # Step temperature at boundaries [K]

nodes1 = []
for i in range(N_nodes):
    if i == 0:
        nodes1.append(Thermostat(label=node_labels[i], temperature=T1, fixed_temperature=T1, material='Al6061'))
    elif i == N_nodes - 1:
        nodes1.append(Thermostat(label=node_labels[i], temperature=T1, fixed_temperature=T1, material='Al6061'))
    else:
        nodes1.append(Node(label=node_labels[i], temperature=T0, material='Al6061'))

# Assign node masses
for i in range(N_nodes):
    m_node = 0.0
    if i > 0:
        m_node += 0.5 * link_masses[i-1]
    if i < N_nodes - 1:
        m_node += 0.5 * link_masses[i]
    nodes1[i].mass = m_node

# Connections
connections1 = []
for i in range(N_nodes - 1):
    connections1.append(Connection(
        node_i=nodes1[i],
        node_j=nodes1[i+1],
        type_='conduction',
        L=L_links[i],
        A=A,
        material='Al6061'
    ))

# Network
net1 = Network(nodes1, connections1, spread=1.0)

# Solve transient
sol1 = net1.solve_ivp_transient(t_span=(0, 200), dt=1.0, method='BDF', verbose=False)

# -----------------------------
# Plots Example 1
# -----------------------------
# T vs time
plt.figure(figsize=(8,5))
for i, label in enumerate(node_labels):
    plt.plot(sol1.t, sol1.y[i,:]-273.15, label=label)
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.title("Example 1: T vs Time (step boundaries)")
plt.legend()
plt.grid(True)
plt.show()

# T vs X at selected times
time_slices1 = [0, 50, 100, 150, 200]
plt.figure(figsize=(8,5))
for t_slice in time_slices1:
    idx = np.argmin(np.abs(sol1.t - t_slice))
    plt.plot(x_positions, sol1.y[:, idx]-273.15, label=f"t={sol1.t[idx]:.0f}s", marker='o')
plt.xlabel("Position [m]")
plt.ylabel("Temperature [°C]")
plt.title("Example 1: T vs Position at different times")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Example 2: Middle Heater
# -----------------------------
nodes2 = []
for i in range(N_nodes):
    if i == 0:
        nodes2.append(Thermostat(label=node_labels[i], temperature=293.15, fixed_temperature=293.15, material='Al6061'))
    elif i == N_nodes - 1:
        nodes2.append(Thermostat(label=node_labels[i], temperature=293.15, fixed_temperature=293.15, material='Al6061'))
    elif i == N_nodes // 2:
        nodes2.append(Heater(label=node_labels[i], temperature=293.15, behaviour_func=100.0, material='Al6061'))
    else:
        nodes2.append(Node(label=node_labels[i], temperature=293.15, material='Al6061'))

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
        type_='conduction',
        L=L_links[i],
        A=A,
        material='Al6061'
    ))

# Network
net2 = Network(nodes2, connections2, spread=1.0)
t_span=(0, 150)
# Solve transient
sol2 = net2.solve_ivp_transient(t_span=t_span, dt=1.5, method='BDF', verbose=False)

# -----------------------------
# Plots Example 2
# -----------------------------
# T vs time
plt.figure(figsize=(8,5))
for i, label in enumerate(node_labels):
    plt.plot(sol2.t, sol2.y[i,:]-273.15, label=label)
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.title("Example 2: T vs Time (middle heater)")
plt.legend()
plt.grid(True)
plt.show()

# T vs X at selected times
time_slices2 = [0, 10, 40, 50, 100, t_span[1]]
plt.figure(figsize=(8,5))
for t_slice in time_slices2:
    idx = np.argmin(np.abs(sol2.t - t_slice))
    plt.plot(x_positions, sol2.y[:, idx]-273.15, label=f"t={sol2.t[idx]:.0f}s", marker='o')
plt.xlabel("Position [m]")
plt.ylabel("Temperature [°C]")
plt.title("Example 2: T vs Position at different times")
plt.legend()
plt.grid(True)
plt.show()


# ==============================================
# Example 3: Radiation Heat transfert case study
# ==============================================
print('\n Radiative examples')
#☺ Sub-Example 3.1  only two plates facing each others
nodes = [
    Thermostat(label='VG1', fixed_temperature=300),
    Thermostat(label='space', fixed_temperature=50)
    ]
connections = [
    Connection(nodes[0], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
    ]
net = Network(nodes, connections)
T, fluxes31, convergence_history = net.solve_steady(verbose=False)
print('Cas3.1')
print(f"{int(T[0])}K -> {int(fluxes31[(nodes[0].label,nodes[1].label)])}W -> {int(T[1])}K")

#☺ Sub-Example 3.2 still two plates facing each others, but a new plate in between, floating shield principle
nodes = [
    Thermostat(label='VG1', fixed_temperature=300),
    Node(label='VG2'),
    Thermostat(label='space', fixed_temperature=50)
    ]
connections = [
    Connection(nodes[0], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
    Connection(nodes[1], nodes[2], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
    ]
net = Network(nodes, connections)
G32,F = net.build_G()
T, fluxes32, convergence_history = net.solve_steady(verbose=False)
print('Cas3.2')
# print(G32)
print(f"{int(T[0])}K -> {int(fluxes32[(nodes[0].label,nodes[1].label)])}W -> {int(T[1])}K -> {int(fluxes32[(nodes[1].label,nodes[2].label)])}W -> {int(T[2])}K")


# #☺ Sub-Example 3.3, should be the same as 3.2
# nodes = [
#     Thermostat(label='VG1', fixed_temperature=300),
#     Node(label='VG2'),
#     Thermostat(label='space', fixed_temperature=50)
#     ]
# connections = [
#     Connection(nodes[0], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
#     Connection(nodes[1], nodes[0], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=0.5),
#     Connection(nodes[1], nodes[2], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=0.5),
#     Connection(nodes[2], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
#     ]
# net = Network(nodes, connections)
# G33,F = net.build_G()
# T, fluxes33, convergence_history = net.solve_steady(verbose=False)
# print('Cas3.3')
# # print(G33)
# print(f"{int(T[0])}K -> {int(fluxes33[(nodes[0].label,nodes[1].label)])}W -> {int(T[1])}K -> {int(fluxes33[(nodes[1].label,nodes[2].label)])}W -> {int(T[2])}K")

#☺ Sub-Example 3.4
nodes = [
    Heater(label='VG1', behaviour_func = 153),
    Node(label='VG2'),
    Thermostat(label='space', fixed_temperature=50)
    ]
connections = [
    Connection(nodes[0], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
    Connection(nodes[1], nodes[2], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
    ]
net = Network(nodes, connections)
T, fluxes34, convergence_history = net.solve_steady(verbose=False)
print('Cas3.4')
print(f"{int(T[0])}K -> {int(fluxes34[(nodes[0].label,nodes[1].label)])}W -> {int(T[1])}K -> {int(fluxes34[(nodes[1].label,nodes[2].label)])}W -> {int(T[2])}K")

# #☺ Sub-Example 3.5 should be the same as 3.4
# nodes = [
#     Heater(label='VG1', behaviour_func = 229),
#     Node(label='VG2'),
#     Thermostat(label='space', fixed_temperature=50)
#     ]
# connections = [
#     Connection(nodes[0], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
#     Connection(nodes[1], nodes[0], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=0.5),
#     Connection(nodes[1], nodes[2], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=0.5),
#     Connection(nodes[2], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
#     ]
# net = Network(nodes, connections)
# T, fluxes35, convergence_history = net.solve_steady(verbose=False)
# print('Cas3.5')
# print(f"{int(T[0])}K -> {int(fluxes35[(nodes[0].label,nodes[1].label)])}W -> {int(T[1])}K -> {int(fluxes35[(nodes[1].label,nodes[2].label)])}W -> {int(T[2])}K")


# #☺ Sub-Example 3.6
# nodes = [
#     Heater(label='VG1', behaviour_func = 229), #0
#     Node(label='VG2_left'), #1
#     Node(label='VG2_right'), #2
#     Thermostat(label='space', fixed_temperature=50) #3
#     ]
# connections = [
#     Connection(nodes[0], nodes[1], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
#     Connection(nodes[1], nodes[0], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
#     Connection(nodes[1], nodes[2], type_='contact', h_c = 1000, A=1),
#     Connection(nodes[3], nodes[2], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1),
#     Connection(nodes[2], nodes[3], type_='radiation', e_i=0.8, e_j=0.8, S_i=1, S_j=1, F_ij=1)
#     ]
# net = Network(nodes, connections)
# T, fluxes36, convergence_history = net.solve_steady(verbose=False)
# print('Cas3.6')
# print(f"From {int(T[0])}K heatflux {int(fluxes36[(nodes[0].label,nodes[1].label)])}W towards {int(T[1])}K")
# print(f"From {int(T[2])}K heatflux {int(fluxes36[(nodes[1].label,nodes[2].label)])}W towards {int(T[3])}K")


# Example 4 : Steady conduction in a large plate
# from : https://drive.uqu.edu.sa/_/kmguedri/files/A-HT-1-Chap5.pdf
nodes = [
    Thermostat(label='0', fixed_temperature = 0+273),
    Heater(label='1', behaviour_func=5e6*10*4e-2), # simplification, lumped internal generation of energy
    Node(label='2'),
    Thermostat(label='inf', fixed_temperature=30+273),
    ]

connections = [
    Connection(nodes[0], nodes[1], type_='conduction',L=2e-2, A=10, material='Uranium'),
    Connection(nodes[1], nodes[2], type_='conduction',L=2e-2, A=10, material='Uranium'),
    Connection(nodes[2], nodes[3], type_='contact', h_c=45, A=10) # air convection
    ]
net = Network(nodes, connections)
T, fluxes36, convergence_history = net.solve_steady(verbose=True)
T2 = T[2] - 273
print(f'Temperature on the external boundary with air is {T2} °C, to compare to analytical 136.0°C')
