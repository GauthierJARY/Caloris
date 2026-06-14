# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:22:58 2026

@author: Admin
"""

import numpy as np
from Caloris.results import SteadyResult


class SteadySolver:

    def __init__(self, network):

        self.network = network

    # ==========================================================
    # Boundary conditions
    # ==========================================================

    def apply_boundary_conditions(
        self,
        G,
        S,
        T,
        fluxes
    ):

        for node in self.network.nodes:

            idx = self.network.node_to_idx[node]

            if node.boundary_type is None:
                continue

            # --------------------------------------
            # Fixed temperature
            # --------------------------------------

            if node.boundary_type == "temperature":
                if node.boundary_function is None:
                    T_fixed = node.boundary_value
                else:
                    incoming_flux = 0.0
                    for (_, dest, _), q in fluxes.items():
                        if dest == node.label:
                            incoming_flux += q
                    
                    T_fixed = node.boundary_function(T[idx], incoming_flux)

                G[idx, :] = 0.0
                G[idx, idx] = 1.0

                S[idx] = T_fixed

            # --------------------------------------
            # Heat input
            # --------------------------------------

            elif node.boundary_type == "heat_input":

                if node.boundary_function is None:

                    S[idx] += node.boundary_value

                else:

                    incoming_flux = 0.0

                    for (_, dest, _), q in fluxes.items():

                        if dest == node.label:
                            incoming_flux += q

                    S[idx] += node.boundary_function(
                        T[idx],
                        incoming_flux
                    )

            else:

                raise ValueError(
                    f"Unknown boundary type "
                    f"{node.boundary_type}"
                )

        return G, S

    # ==========================================================
    # Solve
    # ==========================================================

    def solve(
        self,
        tol=1e-5,
        max_iter=100,
        relaxation=0.5,
        verbose=True
    ):

        T = np.array(
            [
                node.initial_temperature
                for node in self.network.nodes
            ],
            dtype=float
        )

        Q = np.array(
            [
                node.constant_heat_input
                for node in self.network.nodes
            ],
            dtype=float
        )

        convergence_history = []

        for iteration in range(max_iter):

            G = self.network.build_G(T)

            fluxes = self.network.compute_fluxes(T)

            S = Q.copy()

            G_bc, S_bc = self.apply_boundary_conditions(
                G.copy(),
                S.copy(),
                T,
                fluxes
            )

            T_new = np.linalg.solve(
                G_bc,
                S_bc
            )

            T_new = np.maximum(
                T_new,
                0.0
            )

            error = np.max(
                np.abs(T_new - T)
            )

            convergence_history.append(error)

            if error < tol:

                if verbose:

                    print(
                        f"Converged in "
                        f"{iteration+1} iterations."
                    )

                T = T_new

                break

            T = (
                (1-relaxation)*T
                + relaxation*T_new
            )

        return SteadyResult(
            network=self.network,
            T=T,
            fluxes=self.network.compute_fluxes(T),
            G=self.network.build_G(T),
            convergence=convergence_history
        )