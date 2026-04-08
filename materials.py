# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:55:24 2025

@author: Admin
"""
import numpy as np


def lambda_material_dispatch(T, material_conductivity):
    # --- 6061-T6 Aluminum ---
    if material_conductivity == 'Al6061':
        logT = np.log10(T)
        a, b, c, d, e, f, g, h, i = 0.07918, 1.0957, -0.07277, 0.08084, 0.02803, -0.09464, 0.04179, -0.00571, 0.0
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    # --- 6063-T5 Aluminum ---
    elif material_conductivity == 'Al6063':
        logT = np.log10(T)
        a, b, c, d, e, f, g, h, i = 22.401433, -141.13433, 394.95461, -601.15377, 547.83202, -305.99691, 102.38656, -18.810237, 1.4576882
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
        # --- Fonction lambda SST304 ---
    elif material_conductivity == 'SST304':
            logT = np.log10(T)
            a,b,c,d,e,f,g,h,i = -1.4087,1.3982,0.2543,-0.6260,0.2334,0.4256,-0.4658,0.1650,-0.0199
            poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
            return 10**poly
        # --- Fonction lambda SST304L ---
    elif material_conductivity == 'SST304L':
            logT = np.log10(T)
            a,b,c,d,e,f,g,h,i = -1.4087,1.3982,0.2543,-0.6260,0.2334,0.4256,-0.4658,0.1650,-0.0199
            poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
            return 10**poly
        # --- Fonction lambda GFRP ---
    elif material_conductivity == 'CFRP_warp':
            logT = np.log10(T)
            a,b,c,d,e,f,g,h,i = -2.64827, 8.80228, -24.8998, 41.1625, -39.8754, 23.1778, -7.95635, 1.48806, -0.11701
            poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
            return 10**poly
        # --- Fonction lambda Cu RRR=50 ---
    elif material_conductivity == 'Cu_RRR50':
            logT = np.log10(T)
            a,b,c,d,e,f,g,h,i = 1.8743,-0.41538,-0.6018,0.13294, 0.26426, -0.0219, -0.051276, 0.0014871, 0.003723
            poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
            return 10**poly
    elif material_conductivity == 'Cu_RRR10':
        return lambda_material_dispatch(T,'Cu_RRR50')/5
    elif material_conductivity == 'Cu_RRR20':
        return lambda_material_dispatch(T,'Cu_RRR50')/2.5
    elif material_conductivity == 'Cu_RRR100':
        a, b, c, d, e, f, g, h, i = 2.2154, -0.47461, -0.88068, 0.13871, 0.29505, -0.02043, -0.04831, 0.001281, 0.003207
        poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
        return 10**poly
    elif material_conductivity == 'Cu_RRR500':
        a, b, c, d, e, f, g, h, i = 2.8075, -0.54074, -1.2777, 0.15362, 0.36444, -0.02105, -0.051727, 0.0012226, 0.0030964
        poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
        return 10**poly
    elif material_conductivity == 'PEEK':
        # Coefficients a1 à a6
        coefficients = [1.0636976e-1, -1.6340006e-1, 9.4941322e-2, -2.4117988e-2, 2.8797748e-3, -1.3025208e-4 ]
        ln_term = np.log(T + 1)  # ln(T + 1)
        # Calcule A(T) = sum(ai * [ln(T+1)]^i) pour i de 1 à 6
        A_T = sum(coeff * ln_term**(i+1) for i, coeff in enumerate(coefficients))
        A_T = np.maximum(A_T, 0.0108)
        return A_T
    elif material_conductivity == 'Uranium':
        return 28
    else:
        print(f'Warning: No lambda found for {material_conductivity}, default values used !')
        return 5000

def contact_conductance_dispatch(T, G_function_name):
    if G_function_name == 'default':
        return 10.0
    elif G_function_name == 'custom_conductance_200kN':
        """A sample function for direct_G that depends on temperature."""
        # return 0.1 + 0.005 * T
        return 2000
    elif G_function_name == 'custom_conductance_100kN':
        """A sample function for direct_G that depends on temperature."""
        # return 0.1 + 0.005 * T
        return 100
    elif G_function_name == 'custom_conductance_G5':
        """A sample function for direct_G that depends on temperature."""
        return 5
    elif G_function_name == 'custom_conductance_G1':
        """A sample function for direct_G that depends on temperature."""
        return 1
    else:
        print('Warning: No contact parameters found, default values used !')
        return 1.0

def cp_material_dispatch(T, material_specific_heat):
    """
    Returns specific heat [J/kg/K] for a given material as a function of temperature T [K].
    Uses curve-fit polynomial in log10(T) for Al6061, SST304L, Cu_RRR20, or returns default 400.
    """
    T = max(T, 1.0)  # avoid log10(0)
    
    if material_specific_heat == 'SST304L':
        return 385 + 0.02 * (T - 300)
    
    elif material_specific_heat == 'Al6061':
        logT = np.log10(T)
        # coefficients from your table for Cp fit
        a,b,c,d,e,f,g,h,i = 46.6467, -314.292, 866.662, -1298.3, 1162.27, -637.795, 210.351, -38.3094, 2.96344
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    
    elif material_specific_heat == 'Cu_RRR20':
        return 385 + 0.02 * (T - 300)
    
    else:
        return 400  # default fallback
