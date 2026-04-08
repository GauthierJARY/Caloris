# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:57:30 2026

@author: jarygau
"""

import numpy as np

    # --- 6061-T6 Aluminum ---
def conductivity_AL6061(T):
    logT = np.log10(T)
    a, b, c, d, e, f, g, h, i = 0.07918, 1.0957, -0.07277, 0.08084, 0.02803, -0.09464, 0.04179, -0.00571, 0.0
    poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
    return 10**poly

def conductivity_Al6063(T): 
    # --- 6063-T5 Aluminum ---
        logT = np.log10(T)
        a, b, c, d, e, f, g, h, i = 22.401433, -141.13433, 394.95461, -601.15377, 547.83202, -305.99691, 102.38656, -18.810237, 1.4576882
        poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
        return 10**poly
    
def conductivity_SST304L(T):
            logT = np.log10(T)
            a,b,c,d,e,f,g,h,i = -1.4087,1.3982,0.2543,-0.6260,0.2334,0.4256,-0.4658,0.1650,-0.0199
            poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
            return 10**poly
        
def conductivity_CFRP_warp(T):
            logT = np.log10(T)
            a,b,c,d,e,f,g,h,i = -2.64827, 8.80228, -24.8998, 41.1625, -39.8754, 23.1778, -7.95635, 1.48806, -0.11701
            poly = a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8
            return 10**poly
        
def conductivity_CuRRR50(T):
        # --- Fonction lambda Cu RRR=50 ---
            a,b,c,d,e,f,g,h,i = 1.8743,-0.41538,-0.6018,0.13294, 0.26426, -0.0219, -0.051276, 0.0014871, 0.003723
            poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
            return 10**poly

def conductivity_CuRR10(T):
        return conductivity_CuRRR50(T)/5

def conductivity_CuRR20(T):
        return conductivity_CuRRR50(T)/2.5

def conductivity_CuRR100(T):
        a, b, c, d, e, f, g, h, i = 2.2154, -0.47461, -0.88068, 0.13871, 0.29505, -0.02043, -0.04831, 0.001281, 0.003207
        poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
        return 10**poly

def conductivity_CuRR500(T):
        a, b, c, d, e, f, g, h, i = 2.8075, -0.54074, -1.2777, 0.15362, 0.36444, -0.02105, -0.051727, 0.0012226, 0.0030964
        poly = (a + c*T**0.5 + e*T + g*T**1.5 + i*T**2)/(1 + b*T**0.5 + d*T + f*T**1.5 + h*T**2)
        return 10**poly
def conductivity_PEEK(T):
        # Coefficients a1 à a6
        coefficients = [1.0636976e-1, -1.6340006e-1, 9.4941322e-2, -2.4117988e-2, 2.8797748e-3, -1.3025208e-4 ]
        ln_term = np.log(T + 1)  # ln(T + 1)
        # Calcule A(T) = sum(ai * [ln(T+1)]^i) pour i de 1 à 6
        A_T = sum(coeff * ln_term**(i+1) for i, coeff in enumerate(coefficients))
        A_T = np.maximum(A_T, 0.0108)
        return A_T

def conductivity_Uranium(T): 
        return 28

def conductivity_default_material(T):
        print('Warning: No conductivity (W/m/K) found for this material, default values used !')
        return 5000