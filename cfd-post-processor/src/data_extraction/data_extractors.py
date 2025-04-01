"""
Functions for pulling out specific data from CFD results.
"""

import numpy as np
from scipy.interpolate import griddata, interp1d


class DataExtractor:
    """Pulls out specific data from simulation results."""
    
    def __init__(self, data):
        """Set up with simulation data."""
        self.data = data
        
    def extract_line_data(self, start_point, end_point, num_points=100):
        """Gets data along a line between two points."""
        if 'coordinates' not in self.data:
            raise ValueError("No coordinates found in the data")
        
        x = np.linspace(start_point[0], end_point[0], num_points)
        y = np.linspace(start_point[1], end_point[1], num_points)
        
        distance = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
        
        line_data = {
            'coordinates': {
                'x': x,
                'y': y,
                'distance': distance
            }
        }
        
        orig_x = self.data['coordinates']['x']
        orig_y = self.data['coordinates']['y']
        points = np.column_stack((orig_x, orig_y))
        
        for key, value in self.data.items():
            if key != 'coordinates':
                if isinstance(value, dict):
                    line_data[key] = {}
                    for subkey, subvalue in value.items():
                        line_data[key][subkey] = griddata(
                            points, subvalue, (x, y), method='linear'
                        )
                else:
                    line_data[key] = griddata(
                        points, value, (x, y), method='linear'
                    )
        
        return line_data
    
    def extract_surface_data(self, surface_name, x_range=None, y_range=None):
        """Gets data on a specific surface."""
        if 'coordinates' not in self.data:
            raise ValueError("No coordinates found in the data")
        
        orig_x = self.data['coordinates']['x']
        orig_y = self.data['coordinates']['y']
        
        mask = np.ones_like(orig_x, dtype=bool)
        if x_range is not None:
            mask &= (orig_x >= x_range[0]) & (orig_x <= x_range[1])
        if y_range is not None:
            mask &= (orig_y >= y_range[0]) & (orig_y <= y_range[1])
        
        surface_data = {
            'surface_name': surface_name,
            'coordinates': {
                'x': orig_x[mask],
                'y': orig_y[mask]
            }
        }
        
        for key, value in self.data.items():
            if key != 'coordinates':
                if isinstance(value, dict):
                    surface_data[key] = {}
                    for subkey, subvalue in value.items():
                        surface_data[key][subkey] = subvalue[mask]
                else:
                    surface_data[key] = value[mask]
        
        return surface_data
    
    def extract_convergence_history(self):
        """Gets the convergence history data."""
        if 'convergence_history' not in self.data:
            raise ValueError("No convergence history found in the data")
        
        return self.data['convergence_history']
    
    def extract_aerofoil_data(self, x_range=(0, 1)):
        """Gets data along the aerofoil surface."""
        if 'coordinates' not in self.data:
            raise ValueError("No coordinates found in the data")
        
        orig_x = self.data['coordinates']['x']
        orig_y = self.data['coordinates']['y']
        
        mask = (orig_x >= x_range[0]) & (orig_x <= x_range[1])
        x_filtered = orig_x[mask]
        y_filtered = orig_y[mask]
        
        x_unique = np.unique(x_filtered)
        
        x_upper = []
        y_upper = []
        x_lower = []
        y_lower = []
        
        for x in x_unique:
            y_at_x = y_filtered[x_filtered == x]
            if len(y_at_x) > 0:
                y_min = np.min(y_at_x)
                y_max = np.max(y_at_x)
                
                x_upper.append(x)
                y_upper.append(y_max)
                x_lower.append(x)
                y_lower.append(y_min)
        
        upper_idx = np.argsort(x_upper)
        lower_idx = np.argsort(x_lower)
        
        x_upper = np.array(x_upper)[upper_idx]
        y_upper = np.array(y_upper)[upper_idx]
        x_lower = np.array(x_lower)[lower_idx]
        y_lower = np.array(y_lower)[lower_idx]
        
        aerofoil_data = {
            'upper_surface': {
                'coordinates': {
                    'x': x_upper,
                    'y': y_upper
                }
            },
            'lower_surface': {
                'coordinates': {
                    'x': x_lower,
                    'y': y_lower
                }
            }
        }
        
        for key, value in self.data.items():
            if key != 'coordinates':
                if isinstance(value, dict):
                    aerofoil_data['upper_surface'][key] = {}
                    aerofoil_data['lower_surface'][key] = {}
                    
                    for subkey, subvalue in value.items():
                        points = np.column_stack((orig_x, orig_y))
                        aerofoil_data['upper_surface'][key][subkey] = griddata(
                            points, subvalue, (x_upper, y_upper), method='linear'
                        )
                        
                        aerofoil_data['lower_surface'][key][subkey] = griddata(
                            points, subvalue, (x_lower, y_lower), method='linear'
                        )
                else:
                    points = np.column_stack((orig_x, orig_y))
                    aerofoil_data['upper_surface'][key] = griddata(
                        points, value, (x_upper, y_upper), method='linear'
                    )
                    
                    aerofoil_data['lower_surface'][key] = griddata(
                        points, value, (x_lower, y_lower), method='linear'
                    )
        
        return aerofoil_data


class CoordinateUtils:
    """Helper functions for working with coordinates."""
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Figures out the distance between two points."""
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
    
    @staticmethod
    def calculate_normal_distance(point, line_start, line_end):
        """Figures out how far a point is from a line."""
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)
        
        line_vector = line_end - line_start
        point_vector = point - line_start
        
        line_length = np.sqrt(np.sum(line_vector**2))
        
        if line_length == 0:
            return np.sqrt(np.sum(point_vector**2))
        
        line_unit_vector = line_vector / line_length
        
        projection_length = np.dot(point_vector, line_unit_vector)
        projection_point = line_start + projection_length * line_unit_vector
        
        return np.sqrt(np.sum((point - projection_point)**2))
    
    @staticmethod
    def interpolate_along_line(x, y, values, num_points=100):
        """Interpolates values along a line."""
        x = np.array(x)
        y = np.array(y)
        values = np.array(values)
        
        x_interp = np.linspace(x[0], x[-1], num_points)
        y_interp = np.linspace(y[0], y[-1], num_points)
        
        return interp1d(x, values, kind='linear', bounds_error=False, fill_value='extrapolate')(x_interp)
