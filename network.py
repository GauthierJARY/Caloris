# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.

Object-oriented Network class for steady-state thermal solving.
Faithfully reproduces old procedural solver logic.
Caloris Package - Network Module
Handles thermal network topology, structural validation, matrix assembly, and flux calculations.
"""

import numpy as np


class Network:
    """
    Represents the complete thermal network topology composed of nodes and connections.
    """

    def __init__(self, nodes, connections, spread=1.0):
        self.nodes = list(nodes)
        self.connections = list(connections)
        self.spread = float(spread)
        self.N = len(self.nodes)

        # Mapping interne strictement réservé au remplissage des matrices NumPy (0 à N-1),
        # totalement transparent pour l'utilisateur et le debug.
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Exécution des vérifications structurelles à l'initialisation
        self.validate()

    def validate(self):
        """
        Performs rigorous structural validation of the network.
        Raises descriptive errors if inconsistencies or missing references are detected.
        """
        # 1. Vérification de l'unicité des labels de nœuds
        labels = [node.label for node in self.nodes]
        if len(labels) != len(set(labels)):
            duplicates = {label for label in labels if labels.count(label) > 1}
            raise ValueError(
                f"Network Validation Error: Duplicate node labels detected: {duplicates}. "
                f"Every node must have a unique label."
            )

        # 2. Vérification de l'unicité des numéros de nœuds (si renseignés)
        numbers = [node.number for node in self.nodes if node.number is not None]
        if len(numbers) != len(set(numbers)):
            duplicates_num = {num for num in numbers if numbers.count(num) > 1}
            raise ValueError(
                f"Network Validation Error: Duplicate node numbers detected: {duplicates_num}. "
                f"Every node number must be unique."
            )

        # 3. Vérification que chaque connexion relie des nœuds existants dans le réseau
        all_nodes = set(self.nodes)
        for conn in self.connections:
            if conn.node_i not in all_nodes:
                raise ValueError(
                    f"Network Validation Error: Connection '{conn.label}' references source node "
                    f"'{conn.node_i.label}' which is not present in the network nodes list."
                )
            if conn.node_j not in all_nodes:
                raise ValueError(
                    f"Network Validation Error: Connection '{conn.label}' references destination node "
                    f"'{conn.node_j.label}' which is not present in the network nodes list."
                )

            # Vérification de cohérence : une connexion ne doit pas relier un nœud à lui-même
            if conn.node_i == conn.node_j:
                raise ValueError(
                    f"Network Validation Error: Connection '{conn.label}' connects node '{conn.node_i.label}' "
                    f"to itself. Self-connections are not allowed."
                )

    def build_G(self, T):
        """
        Assembles and returns the global conductance matrix G(T) of size (N, N).
        
        Parameters
        ----------
        T : numpy.ndarray
            Current temperature vector across all network nodes [K].
        """
        G = np.zeros((self.N, self.N), dtype=float)

        for conn in self.connections:
            i = self.node_to_idx[conn.node_i]
            j = self.node_to_idx[conn.node_j]

            # Évaluation de la conductance locale dépendant de la température
            G_ij = conn.compute_G(T[i], T[j], spread=self.spread)

            # Assemblage matriciel standard des réseaux nodaux
            G[i, i] += G_ij
            G[j, j] += G_ij
            G[i, j] -= G_ij
            G[j, i] -= G_ij

        return G

    def compute_fluxes(self, T):
        """
        Computes thermal heat fluxes across all connections for a given temperature vector T.
        
        Parameters
        ----------
        T : numpy.ndarray
            Current temperature vector across all network nodes [K].
        
        Returns
        -------
        dict
            Mapping of connection signatures to computed heat fluxes [W].
        """
        fluxes = {}

        for conn in self.connections:
            i = self.node_to_idx[conn.node_i]
            j = self.node_to_idx[conn.node_j]

            G_ij = conn.compute_G(T[i], T[j], spread=self.spread)
            flux = G_ij * (T[i] - T[j])

            key = (conn.node_i.label, conn.node_j.label, conn.connection_type)
            fluxes[key] = flux

        return fluxes
