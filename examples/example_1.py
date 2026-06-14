# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:45:12 2026

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from Caloris.nodes import Node
from Caloris.network import Network
from Caloris.connections import Connection
from Caloris.materials import lambda_material_dispatch
from Caloris.solver import SteadySolver

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
nodes.append(Node(        label='0',
        initial_temperature=300.0,
        specific_heat=None, mass=1.0,
        boundary_type='temperature', boundary_value=T_i, boundary_function=None,
                ))
for i in range(1,n_points):
    x = i * length / n_points
    nodes.append(Node(str(i+1), initial_temperature=4.2 if i == 0 else 300.0))
nodes.append(Node(label=f"{str(n_points+2)}", 
        initial_temperature=300.0,
        specific_heat=None, mass=1.0,
        boundary_type='heat_input', boundary_value=Q_total, boundary_function=None,
                  ))

# Création des connexions
connections = []
for i in range(n_points):
    dx = length / n_points
    connections.append(Connection(nodes[i], nodes[i+1], 'conduction',
                                L=dx, A=area, material_conductivity=material))

# Création du réseau
network = Network(nodes, connections)

# Résolution
solver = SteadySolver(network)
res = solver.solve()
res.check_energy_balance(network)

T = res.T
fluxes = res.fluxes
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