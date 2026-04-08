# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.
"""
class Node:
    def __init__(self, label, temperature=300.0, is_boundary=False, heat_input=0.0, material_specific_heat=None, mass=1e-8):
        """
        Represents a thermal node.
        """
        self.label = label
        self.temperature = temperature
        self.is_boundary = is_boundary
        self.heat_input = heat_input
        self.material_specific_heat = material_specific_heat
        self.mass = mass
    def __repr__(self):
        return f"Node(label={self.label}, T={self.temperature:.2f}K, Q={self.heat_input:.3e}W, material_specific_heat={self.material_specific_heat}, mass={self.mass})"

class Heater(Node):
    def __init__(self, *args, behaviour_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.behaviour = behaviour_func
    def __repr__(self):
        # Using self.behaviour.__name__ is often cleaner than showing the full function object
        func_name = getattr(self.behaviour, '__name__', str(self.behaviour))
        return f"Heater(label={self.label},T={self.temperature:.2f}K, mode={func_name})" 
    
class Cryostat(Node): 
    def __init__(self, *args, behaviour_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.behaviour = behaviour_func
        self.mass = 1e-8 # small mass for transient analysis
    def __repr__(self):
        func_name = getattr(self.behaviour, '__name__', str(self.behaviour))
        return f'Cryostat Node label = {self.label}, T={self.temperature}K, mode={func_name}'

class Thermostat(Node): 
    def __init__(self, *args, fixed_temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.behaviour = fixed_temperature
        self.mass = 1e-8 # small mass for transient analysis
    def __repr__(self):
        func_name = getattr(self.behaviour, '__name__', str(self.behaviour))
        return f'Thermostat Node label = {self.label}, T={self.temperature}K, fixed temperature={func_name}'
