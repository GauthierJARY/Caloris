# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.
"""
# nodes.py

class Node:
    """
    Generic thermal node.

    Parameters
    ----------
    label : str
        User-visible name.

    initial_temperature : float
        Initial guess for steady-state or initial condition for transient.

    constant_heat_input : float
        Constant heat injected into the node [W].

    specific_heat : str or float
        Material specific heat model.

    mass : float
        Thermal mass [kg].

    boundary_type : str
        None
        "temperature"
        "heat_input"

    boundary_value : float
        Value associated with boundary_type.

    boundary_function : callable
        Optional dynamic boundary condition:
            f(T, Q_dot)
    """

    def __init__(self, 
        label,
        initial_temperature=300.0,
        constant_heat_input=0.0,
        specific_heat=None, mass=1.0,
        boundary_type=None, # can be fixed temperature or heat input
        boundary_value=None, boundary_function=None,
                ):

        self.label = str(label)

        self.initial_temperature = float(initial_temperature)

        self.constant_heat_input = float(constant_heat_input)

        self.specific_heat = specific_heat

        self.mass = float(mass)

        self.boundary_type = boundary_type

        self.boundary_value = boundary_value

        self.boundary_function = boundary_function

    def __repr__(self):

        return (
            f"Node("
            f"label='{self.label}', "
            f"T0={self.initial_temperature:.2f} K, "
            f"boundary={self.boundary_type}"
            f")"
        )