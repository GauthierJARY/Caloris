# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025
@author: G.J.
V2 
Created on Fri Oct 23 14:05:42 2026
@author: jarygau
Caloris Package - Integration Test Script
Tests the steady-state resolution of a 3-node thermal network.
"""

from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.steady_solver import SteadySolver

print("--- Initialisation du réseau thermal Caloris ---")

# 1. Création des nœuds
# Nœud 1 : Limite chaude (Température fixée à 350 K)
node_hot = Node(
    label="Electronic_passive",
    number=1,
    node_type="D",
    boundary_value=350.0,
    initial_temperature=350.0
)

# Nœud 2 : Composant interne diffusif (Génère 10 W de chaleur)
node_component = Node(
    label="Electronic_Component",
    number=2,
    node_type="D",
    initial_temperature=300.0,
    qdot=10.0,
)

# Nœud 3 : Limite froide (Température fixée à 200 K)
node_cold = Node(
    label="Cold_Sink",
    number=3,
    node_type="B",
    boundary_type="T",
    boundary_value=200.0,
    initial_temperature=200.0
)

# 2. Création des connexions (Conduction GL)
# Note : Le paramètre 'material' accepte soit un float brut (ex: 200.0 W/m/K), 
# soit un nom de matériau sous forme de chaîne (ex: "aluminium_6061") pour interroger l'Excel.
conn_1 = Connection(
    label="Link_Hot_to_Component",
    node_i=node_hot,
    node_j=node_component,
    connection_type="GL",
    S=0.005,  # Section [m²]
    L=0.05,   # Longueur [m]
    material='Perso'
)

conn_2 = Connection(
    label="Link_Component_to_Cold",
    node_i=node_component,
    node_j=node_cold,
    connection_type="GL",
    S=0.005,
    L=0.05,
    material='Perso'
)

# 3. Assemblage du réseau topologique
nodes = [node_hot, node_component, node_cold]
connections = [conn_1, conn_2]

network = Network(nodes=nodes, connections=connections)
print(f"Réseau assemblé avec succès : {network.N} nœuds, {len(network.connections)} connexions.")

# 4. Configuration et exécution du solveur stationnaire
solver = SteadySolver(network=network)

print("\n--- Lancement de la résolution stationnaire ---")
result = solver.solve(
    tol=1e-6,
    max_iter=100,
    relaxation=0.5,
    verbose=True
)

# 5. Affichage des résultats finaux
print("\n--- RESULTATS DE TEMPERATURES ---")
for node, temp in zip(network.nodes, result.T):
    print(f"Nœud {node.number} ({node.label}) [Type: {node.node_type}] : {temp:.3f} K")

print("\n--- FLUX THERMIQUES PAR LIAISON ---")
for signature, flux in result.fluxes.items():
    node_src, node_dst, c_type = signature
    print(f"Liaison {node_src} -> {node_dst} ({c_type}) : {flux:.3f} W")
     
# Meme example avec l'utilisation de l'appel à Excel

from Caloris.excel import load_network_from_excel,load_solver_config,write_results_to_excel
from Caloris.steady_solver import SteadySolver

excel_path = r"C:\Users\jarygau\Desktop\Caloris\Caloris.xlsx"

print("--- 1. Chargement du réseau depuis Excel ---")
network = load_network_from_excel(excel_path)
print(f"Réseau chargé : {network.N} nœuds, {len(network.connections)} connexions.")

print("--- 2. Chargement des paramètres du solveur ---")
solver_config = load_solver_config(excel_path)

print("--- 3. Exécution de la simulation stationnaire ---")
solver = SteadySolver(network=network)
result = solver.solve(**solver_config)

print("--- 4. Exportation des résultats dans Excel ---")
write_results_to_excel(excel_path, network, result)

print("Process finished successfully.")


