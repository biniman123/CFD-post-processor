"""
File readers for CFD data. Handles different output formats from CFD solvers.
"""

import os
import numpy as np
import pandas as pd


class CFDFileReader:
    """Base class for reading CFD files."""
    
    def __init__(self, file_path):
        """Set up the file reader."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        
    def read(self):
        """Read the file and return the data."""
        raise NotImplementedError("Subclasses must implement read()")
    
    def validate_data(self, data):
        """Check if the data looks valid."""
        return data is not None and len(data) > 0


class FLUENTFileReader(CFDFileReader):
    """Reads ANSYS FLUENT output files."""
    
    def read(self):
        """Read FLUENT data and organize it into a structured format."""
        data = {}
        
        try:
            df = pd.read_csv(self.file_path)
            
            if 'x-coordinate' in df.columns and 'y-coordinate' in df.columns:
                data['coordinates'] = {
                    'x': df['x-coordinate'].values,
                    'y': df['y-coordinate'].values
                }
            
            if 'pressure' in df.columns:
                data['pressure'] = df['pressure'].values
            
            if 'x-velocity' in df.columns and 'y-velocity' in df.columns:
                data['velocity'] = {
                    'x': df['x-velocity'].values,
                    'y': df['y-velocity'].values
                }
                
                data['velocity']['magnitude'] = np.sqrt(
                    df['x-velocity'].values**2 + df['y-velocity'].values**2
                )
            
            if 'turbulent-kinetic-energy' in df.columns:
                data['turbulence'] = {
                    'k': df['turbulent-kinetic-energy'].values
                }
                
                if 'turbulent-dissipation-rate' in df.columns:
                    data['turbulence']['epsilon'] = df['turbulent-dissipation-rate'].values
            
            if 'wall-shear-stress' in df.columns:
                data['wall'] = {
                    'shear_stress': df['wall-shear-stress'].values
                }
                
                if 'y-plus' in df.columns:
                    data['wall']['y_plus'] = df['y-plus'].values
            
            if 'density' in df.columns:
                data['density'] = df['density'].values
            
        except Exception as e:
            print(f"Error reading FLUENT file: {e}")
            return {}
        
        return data


class CSVFileReader(CFDFileReader):
    """Reads generic CSV files with CFD data."""
    
    def __init__(self, file_path, column_mapping=None):
        """Set up the CSV reader with optional column name mapping."""
        super().__init__(file_path)
        self.column_mapping = column_mapping or {}
    
    def read(self):
        """Read CSV data and organize it into a structured format."""
        data = {}
        
        try:
            df = pd.read_csv(self.file_path)
            
            if self.column_mapping:
                df = df.rename(columns=self.column_mapping)
            
            coord_columns = [col for col in df.columns if 'x' in col.lower() or 'y' in col.lower()]
            if coord_columns:
                data['coordinates'] = {}
                for col in coord_columns:
                    data['coordinates'][col] = df[col].values
            
            for col in df.columns:
                if col not in coord_columns:
                    data[col] = df[col].values
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return {}
        
        return data


class ExperimentalDataReader(CFDFileReader):
    """Reads experimental data files for comparison."""
    
    def read(self):
        """Read experimental data and organize it into a structured format."""
        data = {'type': 'experimental'}
        
        try:
            df = pd.read_csv(self.file_path)
            
            if 'x/c' in df.columns and 'Cp' in df.columns:
                data['pressure_coefficient'] = {
                    'x/c': df['x/c'].values,
                    'Cp': df['Cp'].values
                }
            
            if 'y+' in df.columns and 'u+' in df.columns:
                data['boundary_layer'] = {
                    'y_plus': df['y+'].values,
                    'u_plus': df['u+'].values
                }
            
        except Exception as e:
            print(f"Error reading experimental data file: {e}")
            return {}
        
        return data
