# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J. 

Connection class for thermal network.
Handles different types of thermal links: conduction, contact, radiation, direct_G.
"""

import numpy as np
from ThermalNetwork.materials  import lambda_material_dispatch, contact_conductance_dispatch
 
sigma = 5.670374419e-8  # Stefan-Boltzmann constant W/m²K⁴

class Connection:
    def __init__(self, node_i, node_j, type_, **kwargs):
        self.node_i = node_i
        self.node_j = node_j
        self.type = type_  # 'conduction', 'contact', 'radiation', 'direct_G'
        self.params = kwargs  # stores L, A, material, h_c, epsilon, F_ij, G_function_name, etc.
        # Nota Bene: kwargs parameters are handled under dictionnary, so can be inputed and accessed likeso. 
        # kwargs is standard name but any string with ** is working: **duck would work 
    def __repr__(self):
        return f"Link between Node {self.node_i.label} and Node {self.node_j.label} of type {self.type}"
    def compute_G(self, T_i, T_j, spread=1.0): # default spread is 1, that is to say no spread
        """Compute conductance between nodes i and j given temperatures."""
        if self.type == 'conduction':
            L = self.params['L']
            A = self.params['A']
            material = self.params['material']
            T_avg = 0.5 * (abs(T_i) + abs(T_j))
            lambda_avg = lambda_material_dispatch(T_avg, material) * spread
            if lambda_avg <= 0:
                print(f"❌ Zero or negative conductivity for {material} at T_avg={T_avg}")
            G_ij = lambda_avg * A / L

        elif self.type == 'contact':
            A = self.params['A']
            h_c = self.params['h_c']
            G_ij = A * h_c

        elif self.type == 'direct_G':
            G_function_name = self.params['G_function_name']
            T_avg = 0.5 * (abs(T_i) + abs(T_j))
            spread_direct = 1 if spread >= 1 else 0.5
            G_value = contact_conductance_dispatch(T_avg, G_function_name) * spread_direct
            if G_value <= 0:
                print(f"❌ Zero or negative contact conductance for {G_function_name} at T_avg={T_avg}")
            G_ij = G_value * spread_direct

        elif self.type == 'radiation':
            ei = self.params['e_i']
            ej = self.params['e_j']
            Si = self.params['S_i']
            Sj = self.params['S_j']
            F_ij = self.params['F_ij']
            epsilon_eq = 1 / ((1 - ei)/ei + 1/F_ij + Si/Sj * (1 - ej)/ej)
        
            # Stabilzed G_rad computation:
            if abs(T_i - T_j) < 1e-6:
                G_ij = 4 * sigma * epsilon_eq * Si * F_ij * T_i**3 # Taylor order 1 when small differences
            else:
                G_ij = sigma * epsilon_eq * Si * F_ij * (T_i**4 - T_j**4) / (T_i - T_j) # Secante estimation for linearization (taking slope of difference between function(h) - function(a)
                # ref : https://indico.esa.int/event/520/contributions/9876/attachments/6168/11312/Workshop_ESA_Final_Draft.pdf
        else:
            raise ValueError(f"❌ Unknown connection type: {self.type}")

        return G_ij
