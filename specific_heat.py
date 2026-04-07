# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:58:08 2026

@author: jarygau
"""

import numpy as np

def specific_heat_SST304L(T):

def cp_material_dispatch(T, material):
    """
    Returns specific heat [J/kg/K] for a given material as a function of temperature T [K].
    Uses curve-fit polynomial in log10(T) for Al6061, SST304L, Cu_RRR20, or returns default 400.
    """
    T = max(T, 1.0)  # avoid log10(0)
    
    if material == 'SST304L':
        return 385 + 0.02 * (T - 300)
    
    elif material == 'Al6061':
        logT = np.log10(T)
        # coefficients from your table for Cp fit
        a,b,c,d,e,f,g,h,i = 46.6467, -314.292, 866.662, -1298.3, 1162.27, -637.795, 210.351, -38.3094, 2.96344
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    
    elif material == 'Cu_RRR20':
        return 385 + 0.02 * (T - 300)
    
    else:
        return 400  # default fallback
