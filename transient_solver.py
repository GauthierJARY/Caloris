# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:48:15 2026

@author: Admin
"""

import numpy as np
from Caloris.materials import cp_material_dispatch
from Caloris.results import TransientResult


class TransientSolver:

    def __init__(self, network):
        self.network = network

    # ==========================================================
    # Thermal capacitance matrix
    # ==========================================================

    def build_C(self, T):
        C = np.zeros(self.network.N)
        for i, node in enumerate(self.network.nodes):
            if node.specific_heat is None:
                cp = 0.0
            elif isinstance(node.specific_heat, str):
                cp = cp_material_dispatch(
                    T[i],
                    node.specific_heat
                )
            else:
                cp = float(node.specific_heat)
            C[i] = node.mass * cp
        return C

    # ==========================================================
    # Boundary conditions
    # ==========================================================

    def apply_boundary_conditions(
        self,
        A,
        b,
        T_curr,   # Updated to receive current iteration values (T_iter)
        fluxes
    ):
        for node in self.network.nodes:
            idx = self.network.node_to_idx[node]

            if node.boundary_type is None:
                continue

            # ------------------------------------------
            # Fixed temperature boundary
            # ------------------------------------------
            if node.boundary_type == "temperature":
                if node.boundary_function is None:
                    T_fixed = node.boundary_value
                else:
                    # Calculate real-time incoming heat flux for this iteration
                    incoming_flux = 0.0
                    for (_, dest, _), q in fluxes.items():
                        if dest == node.label:
                            incoming_flux += q
                    
                    T_fixed = node.boundary_function(
                        T_curr[idx],
                        incoming_flux
                    )

                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                b[idx] = T_fixed

            # ------------------------------------------
            # Heat input boundary (FIXED: Added missing block)
            # ------------------------------------------
            elif node.boundary_type == "heat_input":
                if node.boundary_function is None:
                    if node.boundary_value is not None:
                        b[idx] += node.boundary_value
                else:
                    # Calculate real-time incoming heat flux for this iteration
                    incoming_flux = 0.0
                    for (_, dest, _), q in fluxes.items():
                        if dest == node.label:
                            incoming_flux += q

                    # Add the dynamic functional heat load directly to the source term
                    b[idx] += node.boundary_function(
                        T_curr[idx],
                        incoming_flux
                    )

            else:
                raise ValueError(
                    f"Unknown boundary type {node.boundary_type}"
                )

        return A, b

    # ==========================================================
    # Solve
    # ==========================================================

    def solve(
        self,
        t_max,
        dt,
        verbose=True
    ):
        n_steps = int(t_max / dt) + 1

        time = np.linspace(
            0,
            t_max,
            n_steps
        )

        T = np.array(
            [
                node.initial_temperature
                for node in self.network.nodes
            ],
            dtype=float
        )

        T_history = np.zeros(
            (
                n_steps,
                self.network.N
            )
        )

        T_history[0] = T

        # ------------------------------------------------------
        # Time loop
        # ------------------------------------------------------
        for n in range(1, n_steps):

            # ----------------------------------------------
            # Nonlinear Picard iterations
            # ----------------------------------------------
            T_iter = T.copy()

            for _ in range(50):
                G = self.network.build_G(T_iter)
                fluxes = self.network.compute_fluxes(T_iter)
                C = self.build_C(T_iter)

                A = G + np.diag(C / dt)

                S = np.array(
                    [
                        node.constant_heat_input
                        for node in self.network.nodes
                    ]
                )

                # b uses previous step thermal inertia (C/dt * T) + base source (S)
                b = S + (C / dt) * T

                # FIXED: Passed T_iter instead of T so boundaries update during Picard loops
                A, b = self.apply_boundary_conditions(
                    A,
                    b,
                    T_iter,
                    fluxes
                )

                T_new = np.linalg.solve(A, b)

                err = np.max(np.abs(T_new - T_iter))
                T_iter = T_new

                if err < 1e-6:
                    break

            T = T_iter
            T_history[n] = T

            if verbose:
                if n % max(1, n_steps // 10) == 0:
                    print(f"{100*n/n_steps:.0f}% complete")

        return TransientResult(
            network=self.network,
            time=time,
            T_history=T_history
        )