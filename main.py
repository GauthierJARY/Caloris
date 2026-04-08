# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.
"""
import numpy as np
from Caloris.network import Network
from Caloris.nodes import Node, Thermostat
from Caloris.connections import Connection


# --- Paramètres du problème ---
L = 1.0         # longueur de la barre [m]
A = 1e-4        # section [m²]
k = 400.0       # conductivité Cuivre [W/mK]
Q = 100.0       # flux appliqué [W]
N = 10          # nombre de nœuds
T0 = 300.0      # température fixée à l'extrémité droite

# --- Positions des nœuds ---
x = np.linspace(0, L, N)

# --- Solution analytique (1D steady-state conduction, flux Q à gauche, T0 à droite) ---
T_analytical = T0 + Q / (k * A) * (x - L)

# --- Création des nœuds pour le solveur OOP ---
nodes = [Node(i, temperature=300.0) for i in range(N)]
nodes[-1] = Thermostat(N-1, fixed_temperature=T0)  # température fixée à droite
nodes[0].heat_input = Q  # flux appliqué à gauche

# --- Connexions entre nœuds consécutifs ---
dx = L / (N-1)
connections = [Connection(nodes[i], nodes[i+1], 'conduction', L=dx, A=A, material_conductivity='Cu') for i in range(N-1)]

# --- Création du réseau ---
network = Network(nodes, connections)

# --- Résolution ---
T_final, fluxes, history = network.solve_steady(tol=1e-8, max_iter=500)

# --- Tableau comparatif ---
print(f"{'Node':>4} | {'Position [m]':>12} | {'T_Analytical [K]':>18} | {'T_Nodal [K]':>12}")
print("-"*60)
for i, node in enumerate(nodes):
    print(f"{node.label:>4} | {x[i]:>12.3f} | {T_analytical[i]:>18.3f} | {node.temperature:>12.3f}")
    
import numpy as np

# Paramètres physiques
L = 1.0        # m
N = 10         # nombre de nœuds
A = 1e-4       # m²
k = 400        # W/mK, cuivre
Q_flux = 100   # W, flux à gauche
T_right = 300  # K, température à droite

dx = L / (N-1)

# --- Création des nœuds et de la matrice G ---
T = np.ones(N) * 300.0
G = np.zeros((N,N))
S = np.zeros(N)

# Assemblage G pour conduction uniforme
for i in range(N-1):
    G[i,i] += k*A/dx
    G[i,i+1] -= k*A/dx
    G[i+1,i] -= k*A/dx
    G[i+1,i+1] += k*A/dx

# Conditions aux limites
# Nœud 0 : flux imposé Q_flux
S[0] = Q_flux

# Nœud N-1 : température fixe T_right
G[-1,:] = 0
G[-1,-1] = 1.0
S[-1] = T_right

# Résolution
T_nodal = np.linalg.solve(G, S)

# Solution analytique (profil linéaire)
x = np.linspace(0,L,N)
T_analytical = T_right + Q_flux*(L-x)/(k*A)

# Affichage tableau
print("Node | Position [m] | T_Analytical [K] | T_Nodal [K]")
print("-----------------------------------------------------")
for i in range(N):
    print(f"{i:4d} | {x[i]:12.3f} | {T_analytical[i]:16.3f} | {T_nodal[i]:12.3f}")
