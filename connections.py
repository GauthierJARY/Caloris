# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J. 
This document is part of the Caloris package.
This script supports the creation of a class & object for the thermal network.
Connection class for thermal network. Handles different types of thermal links: conduction, contact, radiation, direct_G.
Caloris Package - Connection Module
Handles thermal links between nodes: GL (Conduction), GR (Radiation), 
GF (Convection), and GU (User-defined conductance).
"""

from Caloris.materials import property_dispatch

sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W/m²/K⁴]


class Connection:
    """
    Represents a thermal connection between two nodes in the Caloris network.

    Parameters
    ----------
    label : str
        User-visible name for the connection.
    node_i : Node
        First node instance (or node identifier).
    node_j : Node
        Second node instance (or node identifier).
    connection_type : str
        Type of connection: 'GL' (conduction), 'GR' (radiation), 
        'GF' (convection), or 'GU' (user-defined).
    **kwargs : dict
        Specific parameters required depending on the connection_type.
    """

    # Définition des paramètres requis par type de liaison pour la validation
    REQUIRED_PARAMS = {
        "GL": ["S", "L", "material"],
        "GR": ["F_ij"],  # e_i, e_j, S_i, S_j peuvent venir des nœuds ou des kwargs
        "GF": ["A", "h_c"],
        "GU": [],  # Validé dynamiquement (soit G_value, soit G_function_name)
    }

    def __init__(self, label, node_i, node_j, connection_type, **kwargs):
        self.label = str(label)
        self.node_i = node_i
        self.node_j = node_j
        
        self.connection_type = str(connection_type).upper()
        self.params = kwargs

        # Validation rigoureuse à l'initialisation
        self._validate()

    def __repr__(self):
        return (
            f"Connection("
            f"label='{self.label}', "
            f"type='{self.connection_type}', "
            f"nodes=({self.node_i.label} ↔ {self.node_j.label})"
            f")"
        )

    # ============================================================
    # Validation & Sécurité
    # ============================================================

    def _validate(self):
        """Verifies that all required parameters for the given connection type are present and valid."""
        if self.connection_type not in self.REQUIRED_PARAMS:
            raise ValueError(
                f"[Connection '{self.label}']: Unknown connection type '{self.connection_type}'. "
                f"Allowed types are: 'GL' (conduction), 'GR' (radiation), 'GF' (convection), 'GU' (user-defined)."
            )

        # 1. Validation spécifique pour GL (Conduction)
        if self.connection_type == "GL":
            for key in ["S", "L", "material"]:
                if key not in self.params:
                    raise ValueError(
                        f"[Connection '{self.label}' - GL]: Missing mandatory parameter '{key}'. "
                        f"Conduction requires cross-section S, length L, and material (float or str)."
                    )
            if float(self.params["S"]) <= 0:
                raise ValueError(f"[Connection '{self.label}' - GL]: Cross-section S must be strictly positive.")
            if float(self.params["L"]) <= 0:
                raise ValueError(f"[Connection '{self.label}' - GL]: Length L must be strictly positive.")

        # 2. Validation spécifique pour GR (Radiation)
        elif self.connection_type == "GR":
            if "F_ij" not in self.params:
                raise ValueError(
                    f"[Connection '{self.label}' - GR]: Missing mandatory parameter 'F_ij' (view factor)."
                )
            if not (0.0 <= float(self.params["F_ij"]) <= 1.0):
                raise ValueError(f"[Connection '{self.label}' - GR]: View factor F_ij must be between 0 and 1.")

        # 3. Validation spécifique pour GF (Convection)
        elif self.connection_type == "GF":
            for key in ["A", "h_c"]:
                if key not in self.params:
                    raise ValueError(
                        f"[Connection '{self.label}' - GF]: Missing mandatory parameter '{key}'. "
                        f"Convection requires surface area A and convective coefficient h_c (float or str)."
                    )
            if float(self.params["A"]) <= 0:
                raise ValueError(f"[Connection '{self.label}' - GF]: Area A must be strictly positive.")

        # 4. Validation spécifique pour GU (User-defined conductance)
        elif self.connection_type == "GU":
            if "G_value" not in self.params and "G_function_name" not in self.params:
                raise ValueError(
                    f"[Connection '{self.label}' - GU]: User-defined conductance requires either "
                    f"'G_value' (raw float) or 'G_function_name' (str)."
                )

    # ============================================================
    # Public API : Calcul de la Conductance G
    # ============================================================

    def compute_G(self, T_i, T_j, spread=1.0):
        """Computes the thermal conductance G between node i and node j based on temperatures."""
        dispatch = {
            "GL": self._compute_GL,
            "GR": self._compute_GR,
            "GF": self._compute_GF,
            "GU": self._compute_GU,
        }
        return dispatch[self.connection_type](T_i, T_j, spread)

    # ============================================================
    # Private Models (Physique)
    # ============================================================

    def _compute_GL(self, T_i, T_j, spread):
        """Conduction model: G = lambda * S / L"""
        S = float(self.params["S"])
        L = float(self.params["L"])
        material = self.params["material"]
        T_avg = 0.5 * (abs(T_i) + abs(T_j))

        if isinstance(material, (int, float)):
            conductivity = float(material)
        elif isinstance(material, str):
            # Appel générique pour la conductivité thermique ('lambda')
            conductivity = property_dispatch(material, "lambda", T_avg)
        else:
            raise TypeError(f"[Connection '{self.label}']: material must be float or material name string.")

        conductivity *= spread

        if conductivity <= 0:
            raise ValueError(f"[Connection '{self.label}']: Computed conductivity is invalid ({conductivity} W/m/K).")

        return conductivity * S / L

    def _compute_GR(self, T_i, T_j, spread):
        """Radiation model with network view factors and emissivities."""
        F_ij = float(self.params["F_ij"]) * spread

        # Récupération des émissivités et surfaces (priorité aux kwargs de la liaison, sinon attributs des nœuds si dispos)
        e_i = self.params.get("e_i", getattr(self.node_i, "emissivity", 1.0))
        e_j = self.params.get("e_j", getattr(self.node_j, "emissivity", 1.0))
        S_i = self.params.get("S_i", getattr(self.node_i, "surface", 1.0))
        S_j = self.params.get("S_j", getattr(self.node_j, "surface", 1.0))

        # Facteur d'émissivité équivalent (surfaces parallèles / enceintes)
        epsilon_eq = 1.0 / (
            (1.0 - e_i) / e_i
            + 1.0 / F_ij
            + (S_i / S_j) * ((1.0 - e_j) / e_j)
        )

        # Linéarisation ou calcul direct du flux radiatif converti en conductance équivalente G_rad
        if abs(T_i - T_j) < 1e-6:
            return 4.0 * sigma * epsilon_eq * S_i * F_ij * (T_i ** 3)

        return (
            sigma
            * epsilon_eq
            * S_i
            * F_ij
            * (T_i ** 4 - T_j ** 4)
            / (T_i - T_j)
        )

    def _compute_GF(self, T_i, T_j, spread):
        """Convection model: G = A * h_c"""
        A = float(self.params["A"])
        h_c_input = self.params["h_c"]
        T_avg = 0.5 * (abs(T_i) + abs(T_j))

        if isinstance(h_c_input, (int, float)):
            h_c = float(h_c_input)
        elif isinstance(h_c_input, str):
            # Appel générique pour le coefficient de convection ('h_c')
            h_c = property_dispatch(h_c_input, "h_c", T_avg)
        else:
            raise TypeError(f"[Connection '{self.label}']: h_c must be a float or a model name string.")

        return A * h_c * spread

    def _compute_GU(self, T_i, T_j, spread):
        """User-defined conductance model (fixed float or function lookup)."""
        if "G_value" in self.params:
            return float(self.params["G_value"]) * spread

        model_name = self.params["G_function_name"]
        T_avg = 0.5 * (abs(T_i) + abs(T_j))

        # Appel générique pour une conductance utilisateur ('conductance' ou 'G')
        G = property_dispatch(model_name, "conductance", T_avg)
        return G * spread
