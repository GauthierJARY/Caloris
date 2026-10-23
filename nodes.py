# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.
"""
class Node:
    """
    Generic thermal node.

    Parameters
    ----------
    label : str
        User-visible name.

    group_number : int or str
        Group number for node organization.

    node_type : str
        Node classification: 'D' for diffusive or 'B' for boundary.

    initial_temperature : float
        Initial temperature T0 for steady-state guess or transient condition [K].

    qdot : float
        Constant heat load injected into the node [W].

    mass : float
        Mass of the node [kg].

    capacitance : float or str
        Thermal capacitance value [J/K] or material name (string) for property lookups.

    boundary_type : str, optional
        Type of boundary condition (required if node_type == 'B'): 'T' or 'Qdot'.

    boundary_value : float, optional
        Value associated with boundary_type (required if node_type == 'B').
    """

    def __init__(
        self,
        label,
        number=None,
        group_number=0,
        node_type="D",
        initial_temperature=300.0,
        qdot=0.0,
        mass=1.0,
        capacitance=1.0,
        boundary_type=None,
        boundary_value=None,
    ):
        self.label = str(label)
        self.number = int(number) if number is not None else None
        self.group_number = group_number
        
        # Validation of node type ('D' ou 'B')
        node_type = str(node_type).upper()
        if node_type not in ("D", "B"):
            raise ValueError("node_type must be 'D' (diffusive) or 'B' (boundary).")
        self.node_type = node_type

        self.initial_temperature = float(initial_temperature)
        self.qdot = float(qdot)
        self.mass = float(mass)
        self.capacitance = capacitance

        # Manipulation of boundaries
        if self.node_type == "B":
            if boundary_type is None:
                raise ValueError(f"Node '{self.label}' is of type 'B' (boundary) and requires a boundary_type ('T' fixed temperature or 'Qdot' fixed heatload).")
            
            bt_upper = str(boundary_type).upper()
            if bt_upper == "T":
                self.boundary_type = "T"
            elif bt_upper in ("QDOT", "Q_DOT"):
                self.boundary_type = "Qdot"
            else:
                raise ValueError("boundary_type must be either 'T' (fixed temperature) or 'Qdot' (constant heat input).")
            
            if boundary_value is None:
                raise ValueError(f"Node '{self.label}' with boundary_type '{self.boundary_type}' requires a boundary_value.")
            self.boundary_value = float(boundary_value)
        else:
            # Pour un nœud diffusif ('D'), les frontières ne doivent pas être renseignées
            if boundary_type is not None:
                raise ValueError(f"Node '{self.label}' is of type 'D' (diffusive) and cannot have a boundary_type.")
            self.boundary_type = None
            self.boundary_value = None

        def __repr__(self):
                num_str = f"number={self.number}, " if self.number is not None else ""
                bound_str = f", boundary={self.boundary_type}:{self.boundary_value}" if self.node_type == "B" else ""
                return (
                    f"Node("
                    f"label='{self.label}', "
                    f"{num_str}"
                    f"type='{self.node_type}', "
                    f"T0={self.initial_temperature:.2f}K"
                    f"{bound_str}"
                    f")"
                )
