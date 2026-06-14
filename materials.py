# -*- coding: utf-8 -*-
"""
Created on Dynamic Excel Framework Migration

@author: Admin
"""
# -*- coding: utf-8 -*-
"""
Created on Dynamic Excel Framework Migration
Modified for Case-Insensitive Lookups

@author: Admin
"""
import os
import pandas as pd
from scipy.interpolate import interp1d

class MaterialDatabase:
    def __init__(self, excel_path="material_properties.xlsx"):
        """
        Loads the Excel properties database once upon framework initialization.
        Compiles linear interpolation curves for every physical property found.
        """
        if not os.path.exists(excel_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(base_dir, excel_path)
            
        if not os.path.exists(excel_path):
            raise FileNotFoundError(
                f"Lumped Thermal Solver Error: Could not locate database sheet at: '{excel_path}'"
            )
            
        self.raw_data = pd.read_excel(excel_path, sheet_name=None)
        self.interpolators = {}
        self._build_interpolators()

    def _build_interpolators(self):
        """Processes each Excel sheet and maps columns to interpolation objects."""
        for mat_name, df in self.raw_data.items():
            # Standardize sheet names to lowercase to prevent case mismatch warnings
            mat_key = str(mat_name).strip().lower()
            
            # Clean column headers to eliminate accidental spaces or casing errors
            df.columns = [str(col).strip().lower() for col in df.columns]
            
            # Find the temperature column (e.g., 'temperature (k)' or 't')
            t_col = [col for col in df.columns if 'temperature' in col or col == 't']
            if not t_col:
                print(f"Warning: No temperature column found in sheet '{mat_name}'. Skipping.")
                continue
                
            T_data = df[t_col[0]].values
            self.interpolators[mat_key] = {}
            
            # Loop through all other columns (lambda, cp, rho, etc.)
            for col in df.columns:
                if col == t_col[0]:
                    continue
                
                prop_values = df[col].values
                # Linear interpolation with bounds safety extrapolation
                self.interpolators[mat_key][col] = interp1d(
                    T_data, prop_values, kind='linear', bounds_error=False, fill_value="extrapolate"
                )

    def query(self, material, prop_name, temperature, fallback_value):
        """Queries the sheet interpolation matrix using case-insensitive keys."""
        mat_key = str(material).strip().lower()
        prop_key = str(prop_name).strip().lower()
        
        if mat_key not in self.interpolators or prop_key not in self.interpolators[mat_key]:
            print(f"Warning: No '{prop_key}' found for '{material}' in Excel. Using fallback: {fallback_value}")
            return fallback_value
            
        return float(self.interpolators[mat_key][prop_key](temperature))


# Initialize the database instance EXACTLY ONCE when this module is imported
_db = MaterialDatabase("material_properties.xlsx")


# ============================================================
# Public API Wrappers (Keeps your Connection class happy)
# ============================================================

def lambda_material_dispatch(T, material_conductivity):
    """Returns thermal conductivity [W/m/K] from Excel."""
    return _db.query(material_conductivity, "lambda", T, fallback_value=5000.0)


def cp_material_dispatch(T, material_specific_heat):
    """Returns specific heat [J/kg/K] from Excel."""
    return _db.query(material_specific_heat, "cp", T, fallback_value=400.0)


def contact_conductance_dispatch(T, G_function_name):
    """Returns contact conductance from Excel."""
    return _db.query(G_function_name, "conductance", T, fallback_value=1.0)