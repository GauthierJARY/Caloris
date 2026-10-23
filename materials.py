# -*- coding: utf-8 -*-
"""
Created on Dynamic Excel Framework Migration
Created on Dynamic Excel Framework Migration
Modified for Case-Insensitive Lookups
Caloris Package - Materials & Dynamic Properties Module
Generic database lookup and interpolation engine using Excel files.

@GJ
"""

import os
import pandas as pd
from scipy.interpolate import interp1d


class PropertyDatabase:
    """
    Loads an Excel properties database and handles linear interpolation 
    for materials or user-defined function sheets.
    """

    def __init__(self, excel_path="material_properties.xlsx"):
        if not os.path.exists(excel_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(base_dir, excel_path)

        if not os.path.exists(excel_path):
            raise FileNotFoundError(
                f"[Caloris Database Error]: Could not locate property Excel database at: '{excel_path}'"
            )

        self.excel_path = excel_path
        self.raw_data = pd.read_excel(excel_path, sheet_name=None)
        self.interpolators = {}
        self._build_interpolators()

    def _build_interpolators(self):
        """Processes each sheet and maps every property column to a 1D interpolator."""
        for sheet_name, df in self.raw_data.items():
            sheet_key = str(sheet_name).strip().lower()

            # Normalisation des entêtes de colonnes (minuscules sans espaces parasites)
            df.columns = [str(col).strip().lower() for col in df.columns]

            # Recherche de la colonne de température (ex: 'temperature', 't', 'temp', 'temperature (k)')
            t_col = [col for col in df.columns if "temp" in col or col == "t"]
            if not t_col:
                print(f"Warning: No temperature column found in sheet '{sheet_name}'. Skipping.")
                continue

            t_col_name = t_col[0]
            T_data = df[t_col_name].values
            self.interpolators[sheet_key] = {}

            # Création d'un interpolateur pour chaque propriété présente dans la feuille
            for col in df.columns:
                if col == t_col_name:
                    continue

                prop_values = df[col].values
                # Interpolation linéaire avec extrapolation de sécurité sur les bords
                self.interpolators[sheet_key][col] = interp1d(
                    T_data,
                    prop_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

    def query(self, sheet_name, prop_name, temperature, fallback_value=None):
        """
        Generic query engine.
        
        Parameters
        ----------
        sheet_name : str
            Name of the Excel sheet (material name or custom function name).
        prop_name : str
            Name of the property column (e.g. 'lambda', 'cp', 'conductance', 'h_c').
        temperature : float
            Temperature at which to evaluate the property [K].
        fallback_value : float, optional
            Default value if key/sheet is missing. Raises ValueError if None and missing.
        """
        sheet_key = str(sheet_name).strip().lower()
        prop_key = str(prop_name).strip().lower()

        # 1. Vérification de la présence de la feuille
        if sheet_key not in self.interpolators:
            if fallback_value is not None:
                print(f"Warning: Sheet '{sheet_name}' not found in Excel. Using fallback: {fallback_value}")
                return float(fallback_value)
            raise KeyError(
                f"[PropertyDatabase Error]: Sheet '{sheet_name}' not found in '{self.excel_path}'. "
                f"Available sheets: {list(self.interpolators.keys())}"
            )

        # 2. Vérification de la présence de la propriété/colonne
        if prop_key not in self.interpolators[sheet_key]:
            # Fallback direct : si la feuille ne contient qu'une seule colonne de donnée, on la prend par défaut
            available_props = list(self.interpolators[sheet_key].keys())
            if len(available_props) == 1:
                prop_key = available_props[0]
            elif fallback_value is not None:
                print(f"Warning: Property '{prop_name}' not found in sheet '{sheet_name}'. Using fallback: {fallback_value}")
                return float(fallback_value)
            else:
                raise KeyError(
                    f"[PropertyDatabase Error]: Property '{prop_name}' not found in sheet '{sheet_name}'. "
                    f"Available properties in this sheet: {available_props}"
                )

        return float(self.interpolators[sheet_key][prop_key](temperature))


# Initialisation unique à l'importation du package
_db = PropertyDatabase("material_properties.xlsx")


# ============================================================
# API Publique Généraliste (Utilisée par Connection et Node)
# ============================================================

def property_dispatch(sheet_name, prop_name, T, fallback_value=None):
    """
    Universal dispatch function for Caloris.
    Evaluates any material property or user-defined function from Excel.

    Examples
    --------
    >>> property_dispatch("Aluminium_6061", "lambda", T=100.0)
    >>> property_dispatch("My_Contact_Law", "conductance", T=100.0)
    >>> property_dispatch("Convection_Model", "h_c", T=300.0)
    """
    return _db.query(sheet_name, prop_name, T, fallback_value=fallback_value)
