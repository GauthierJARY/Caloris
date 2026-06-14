# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J. 
This document is part of the Caloris package.
This script supports the creation of a class & object for the thermal network.
Connection class for thermal network. Handles different types of thermal links: conduction, contact, radiation, direct_G.
"""

from Caloris.materials import (
    lambda_material_dispatch,
    contact_conductance_dispatch
)

sigma = 5.670374419e-8  # W/m²/K⁴


class Connection:

    REQUIRED_PARAMS = {
        "conduction": ["L", "A", "material_conductivity"],
        "convection": ["A", "h_c"],
        "conductance": ["G_function_name"],
        "radiation": ["e_i", "e_j", "S_i", "S_j", "F_ij"],
    }

    def __init__(self, node_i, node_j, connection_type, **kwargs):

        self.node_i = node_i
        self.node_j = node_j

        self.connection_type = connection_type.lower()

        self.params = kwargs

        self._validate()

    def __repr__(self):

        return (
            f"Connection("
            f"{self.node_i.label} ↔ {self.node_j.label}, "
            f"type='{self.connection_type}')"
        )

    def _validate(self):

        if self.connection_type not in self.REQUIRED_PARAMS:
            raise ValueError(
                f"Unknown connection type '{self.connection_type}'"
            )

        required = self.REQUIRED_PARAMS[self.connection_type]

        for key in required:
            if key not in self.params:
                raise ValueError(
                    f"Missing parameter '{key}' "
                    f"for {self.connection_type} connection"
                )

    # ============================================================
    # Public API
    # ============================================================

    def compute_G(self, T_i, T_j, spread=1.0):

        dispatch = {
            "conduction": self._compute_conduction_G,
            "convection": self._compute_convection_G,
            "conductance": self._compute_contact_G,
            "radiation": self._compute_radiation_G,
        }

        return dispatch[self.connection_type](T_i, T_j, spread)

    # ============================================================
    # Private models
    # ============================================================

    def _compute_conduction_G(self, T_i, T_j, spread):

        L = self.params["L"]
        A = self.params["A"]
        material = self.params["material_conductivity"]

        T_avg = 0.5 * (abs(T_i) + abs(T_j))

        if isinstance(material, (int, float)):
            conductivity = float(material)

        elif isinstance(material, str):
            conductivity = lambda_material_dispatch(
                T_avg,
                material
            )

        else:
            raise TypeError(
                "material_conductivity must be "
                "float or material name string"
            )

        conductivity *= spread

        if conductivity <= 0:
            raise ValueError(
                f"Invalid conductivity ({conductivity})"
            )

        return conductivity * A / L

    def _compute_convection_G(self, T_i, T_j, spread):

        A = self.params["A"]
        h_c = self.params["h_c"]

        return A * h_c

    def _compute_contact_G(self, T_i, T_j, spread):

        model = self.params["G_function_name"]

        T_avg = 0.5 * (abs(T_i) + abs(T_j))

        spread_factor = 1.0 if spread >= 1 else 0.5

        G = (
            contact_conductance_dispatch(
                T_avg,
                model
            )
            * spread_factor
        )

        if G <= 0:
            raise ValueError(
                f"Invalid contact conductance ({G})"
            )

        return G

    def _compute_radiation_G(self, T_i, T_j, spread):

        ei = self.params["e_i"]
        ej = self.params["e_j"]

        Si = self.params["S_i"]
        Sj = self.params["S_j"]

        F_ij = self.params["F_ij"]

        epsilon_eq = 1.0 / (
            (1 - ei) / ei
            + 1 / F_ij
            + Si / Sj * (1 - ej) / ej
        )

        if abs(T_i - T_j) < 1e-6:

            return (
                4
                * sigma
                * epsilon_eq
                * Si
                * F_ij
                * T_i**3
            )

        return (
            sigma
            * epsilon_eq
            * Si
            * F_ij
            * (T_i**4 - T_j**4)
            / (T_i - T_j)
        )