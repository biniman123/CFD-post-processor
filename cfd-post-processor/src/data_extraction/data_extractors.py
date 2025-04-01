"""
Data Extractors Module

This module contains classes and functions for extracting specific data from CFD simulation results.
"""

import numpy as np
from scipy.interpolate import griddata, interp1d


class DataExtractor:
    """Extract specific data from simulation results.
    
    This class provides methods to extract data along lines, on surfaces,
    and from convergence history for further analysis.
    
    Attributes:
        data (dict): Dictionary containing the simulation data
    """
    
    def __init__(self, data):
        """Initialize the data extractor with simulation data.
        
        Args:
            data (dict): Dictionary containing the simulation data
        """
        self.data = data
        
    def extract_line_data(self, start_point, end_point, num_points=100):
        """Extract data along a line between two points.
        
        Args:
            start_point (tuple): Starting point coordinates (x, y)
            end_point (tuple): Ending point coordinates (x, y)
            num_points (int, optional): Number of points to extract along the line
        
        Returns:
            dict: Dictionary containing the extracted data along the line
        """
        # Check if coordinates exist in the data
        if 'coordinates' not in self.data:
            raise ValueError("Coordinates not found in the data")
        
        # Generate points along the line
        x = np.linspace(start_point[0], end_point[0], num_points)
        y = np.linspace(start_point[1], end_point[1], num_points)
        
        # Calculate distance along the line
        distance = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
        
        # Extract data at each point using interpolation
        line_data = {
            'coordinates': {
                'x': x,
                'y': y,
                'distance': distance
            }
        }
        
        # Get the original coordinates
        orig_x = self.data['coordinates']['x']
        orig_y = self.data['coordinates']['y']
        points = np.column_stack((orig_x, orig_y))
        
        # Interpolate each data field along the line
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
        """Extract data on a specific surface.
        
        Args:
            surface_name (str): Name of the surface to extract data from
            x_range (tuple, optional): Range of x-coordinates to extract (min, max)
            y_range (tuple, optional): Range of y-coordinates to extract (min, max)
        
        Returns:
            dict: Dictionary containing the extracted data on the surface
        """
        # This is a simplified implementation - actual implementation would depend
        # on how surfaces are defined in the simulation data
        
        # For demonstration purposes, we'll assume surfaces are defined by coordinate ranges
        if 'coordinates' not in self.data:
            raise ValueError("Coordinates not found in the data")
        
        # Get the original coordinates
        orig_x = self.data['coordinates']['x']
        orig_y = self.data['coordinates']['y']
        
        # Apply coordinate ranges if provided
        mask = np.ones_like(orig_x, dtype=bool)
        if x_range is not None:
            mask &= (orig_x >= x_range[0]) & (orig_x <= x_range[1])
        if y_range is not None:
            mask &= (orig_y >= y_range[0]) & (orig_y <= y_range[1])
        
        # Extract data on the surface
        surface_data = {
            'surface_name': surface_name,
            'coordinates': {
                'x': orig_x[mask],
                'y': orig_y[mask]
            }
        }
        
        # Extract each data field on the surface
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
        """Extract convergence history data.
        
        Returns:
            dict: Dictionary containing the convergence history data
        """
        # This is a simplified implementation - actual implementation would depend
        # on how convergence history is stored in the simulation data
        
        if 'convergence_history' not in self.data:
            raise ValueError("Convergence history not found in the data")
        
        return self.data['convergence_history']
    
    def extract_aerofoil_data(self, x_range=(0, 1)):
        """Extract data along the aerofoil surface.
        
        Args:
            x_range (tuple, optional): Range of x-coordinates to extract (min, max)
        
        Returns:
            dict: Dictionary containing the extracted data along the aerofoil surface
        """
        # This is a simplified implementation - actual implementation would depend
        # on how the aerofoil surface is defined in the simulation data
        
        # For demonstration purposes, we'll assume the aerofoil surface is defined
        # by points with the minimum and maximum y-coordinates at each x-coordinate
        
        if 'coordinates' not in self.data:
            raise ValueError("Coordinates not found in the data")
        
        # Get the original coordinates
        orig_x = self.data['coordinates']['x']
        orig_y = self.data['coordinates']['y']
        
        # Apply x-coordinate range
        mask = (orig_x >= x_range[0]) & (orig_x <= x_range[1])
        x_filtered = orig_x[mask]
        y_filtered = orig_y[mask]
        
        # Find unique x-coordinates
        x_unique = np.unique(x_filtered)
        
        # Initialize arrays for upper and lower surfaces
        x_upper = []
        y_upper = []
        x_lower = []
        y_lower = []
        
        # For each unique x-coordinate, find the points with minimum and maximum y-coordinates
        for x in x_unique:
            y_at_x = y_filtered[x_filtered == x]
            if len(y_at_x) > 0:
                y_min = np.min(y_at_x)
                y_max = np.max(y_at_x)
                
                x_upper.append(x)
                y_upper.append(y_max)
                x_lower.append(x)
                y_lower.append(y_min)
        
        # Sort the points by x-coordinate
        upper_idx = np.argsort(x_upper)
        lower_idx = np.argsort(x_lower)
        
        x_upper = np.array(x_upper)[upper_idx]
        y_upper = np.array(y_upper)[upper_idx]
        x_lower = np.array(x_lower)[lower_idx]
        y_lower = np.array(y_lower)[lower_idx]
        
        # Extract data for upper and lower surfaces
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
        
        # Extract each data field on the aerofoil surface
        for key, value in self.data.items():
            if key != 'coordinates':
                if isinstance(value, dict):
                    # Upper surface
                    aerofoil_data['upper_surface'][key] = {}
                    # Lower surface
                    aerofoil_data['lower_surface'][key] = {}
                    
                    for subkey, subvalue in value.items():
                        # Interpolate data for upper surface
                        points = np.column_stack((orig_x, orig_y))
                        aerofoil_data['upper_surface'][key][subkey] = griddata(
                            points, subvalue, (x_upper, y_upper), method='linear'
                        )
                        
                        # Interpolate data for lower surface
                        aerofoil_data['lower_surface'][key][subkey] = griddata(
                            points, subvalue, (x_lower, y_lower), method='linear'
                        )
                else:
                    # Interpolate data for upper surface
                    points = np.column_stack((orig_x, orig_y))
                    aerofoil_data['upper_surface'][key] = griddata(
                        points, value, (x_upper, y_upper), method='linear'
                    )
                    
                    # Interpolate data for lower surface
                    aerofoil_data['lower_surface'][key] = griddata(
                        points, value, (x_lower, y_lower), method='linear'
                    )
        
        return aerofoil_data


class CoordinateUtils:
    """Utilities for handling coordinate systems in CFD simulations.
    
    This class provides methods for coordinate transformations, distance calculations,
    and other coordinate-related operations.
    """
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate the Euclidean distance between two points.
        
        Args:
            point1 (tuple): First point coordinates (x, y)
            point2 (tuple): Second point coordinates (x, y)
        
        Returns:
            float: Euclidean distance between the points
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    @staticmethod
    def calculate_normal_distance(point, line_start, line_end):
        """Calculate the normal distance from a point to a line.
        
        Args:
            point (tuple): Point coordinates (x, y)
            line_start (tuple): Line starting point coordinates (x, y)
            line_end (tuple): Line ending point coordinates (x, y)
        
        Returns:
            float: Normal distance from the point to the line
        """
        # Calculate the line vector
        line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
        
        # Calculate the point vector
        point_vec = (point[0] - line_start[0], point[1] - line_start[1])
        
        # Calculate the line length
        line_length = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
        
        # Calculate the normal distance
        normal_distance = abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]) / line_length
        
        return normal_distance
    
    @staticmethod
    def interpolate_along_line(x, y, values, num_points=100):
        """Interpolate values along a line defined by x and y coordinates.
        
        Args:
            x (array): x-coordinates of the line
            y (array): y-coordinates of the line
            values (array): Values to interpolate
            num_points (int, optional): Number of points for interpolation
        
        Returns:
            tuple: Tuple containing (distances, interpolated_values)
        """
        # Calculate distances along the line
        distances = np.zeros_like(x)
        for i in range(1, len(x)):
            distances[i] = distances[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        
        # Create interpolation function
        interp_func = interp1d(distances, values, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Generate evenly spaced points along the line
        new_distances = np.linspace(distances[0], distances[-1], num_points)
        
        # Interpolate values at the new points
        new_values = interp_func(new_distances)
        
        return new_distances, new_values
