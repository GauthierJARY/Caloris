# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:14:03 2026

@author: jarygau
"""

def conductance_default(T):
    print('Warning: No contact parameters found, default values used !')
    return 1

def conductance_custom_5(T):
    return 5

def conductance_custom_200kN(T):
    return 2000
