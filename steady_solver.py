# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:22:58 2026

@author: GJ
Caloris Package - Steady Solver Module
Handles steady-state thermal network resolution using iterative linearization.
"""

import numpy as np
from Caloris.results import SteadyResult


class SteadySolver:
    """
    Performs steady-state thermal simulation for a given network.
    """

    def __init__(self, network):
        self.network = network

    def apply_boundary_conditions(self, G, S, T):
        """
        Applies Dirichlet (fixed T) and Neumann (fixed Qdot) boundary conditions 
        based on the node architecture ('D' vs 'B').
        """
        for node in self.network.nodes:
            idx = self.network.node_to_idx[node]

            # Diffusive nodes do not have direct boundary constraints applied here
            if node.node_type != "B":
                continue

            # --------------------------------------
            # Fixed Temperature Boundary ('T')
            # --------------------------------------
            if node.boundary_type == "T":
                T_fixed = float(node.boundary_value)
                
                # Zero out the row in G, set diagonal to 1.0, and set S to T_fixed
                G[idx, :] = 0.0
                G[idx, idx] = 1.0
                S[idx] = T_fixed

            # --------------------------------------
            # Constant Heat Input Boundary ('Qdot')
            # --------------------------------------
            elif node.boundary_type == "Qdot":
                S[idx] += float(node.boundary_value)

            else:
                raise ValueError(
                    f"SteadySolver Error: Unknown boundary_type '{node.boundary_type}' for node '{node.label}'."
                )

        return G, S

    def solve(
        self,
        tol=1e-5,
        max_iter=100,
        relaxation=0.5,
        verbose=True
    ):
        """
        Solves the non-linear thermal network system iteratively.
        """
        # Initial temperature vector from nodes' initial_temperature (T0)
        T = np.array(
            [node.initial_temperature for node in self.network.nodes],
            dtype=float
        )

        # Base internal heat sources from nodes' qdot attribute
        Q_sources = np.array(
            [node.qdot for node in self.network.nodes],
            dtype=float
        )

        convergence_history = []

        for iteration in range(max_iter):
            # 1. Build global conductance matrix G(T)
            G = self.network.build_G(T)

            # 2. Copy base heat sources vector
            S = Q_sources.copy()

            # 3. Apply boundary conditions (modifies G and S)
            G_bc, S_bc = self.apply_boundary_conditions(
                G.copy(),
                S.copy(),
                T
            )

            # 4. Solve linear system G_bc * T_new = S_bc
            try:
                T_new = np.linalg.solve(G_bc, S_bc)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"SteadySolver Error: Linear solver failed at iteration {iteration+1}. "
                    f"The conductance matrix might be singular or ill-conditioned. Details: {e}"
                )

            # Physical safety bound: no absolute sub-zero Kelvin temperatures
            T_new = np.maximum(T_new, 0.0)

            # 5. Compute maximum convergence error (L-infinity norm)
            error = np.max(np.abs(T_new - T))
            convergence_history.append(error)

            if verbose and (iteration % 10 == 0 or error < tol):
                print(f"Iteration {iteration+1:3d} | Max Temp Error: {error:.4e} K")

            if error < tol:
                if verbose:
                    print(f"SteadySolver: Converged successfully in {iteration+1} iterations.")
                T = T_new
                break

            # 6. Under-relaxation to stabilize convergence for non-linear effects (e.g. radiation)
            T = (1.0 - relaxation) * T + relaxation * T_new
        else:
            if verbose:
                print(f"SteadySolver Warning: Maximum iterations ({max_iter}) reached without full convergence (Error: {error:.4e}).")

        # 7. Return complete results object
        return SteadyResult(
            network=self.network,
            T=T,
            fluxes=self.network.compute_fluxes(T),
            G=self.network.build_G(T),
            convergence=convergence_history
        )
