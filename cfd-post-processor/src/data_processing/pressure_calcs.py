"""
Pressure Calculations Module

This module contains functions for calculating pressure-related parameters
such as pressure coefficient and pressure distributions. These calculations
are particularly relevant for aerodynamic analysis of F1 components.
"""

import numpy as np


def calculate_pressure_coefficient(pressure, reference_pressure, dynamic_pressure):
    """Calculate pressure coefficient from pressure values.
    
    The pressure coefficient is defined as:
    Cp = (p - p_ref) / q
    
    This is one of the most important parameters in aerodynamics, as it allows
    for comparison between different flow conditions and geometries.
    
    Args:
        pressure (float or array): Local static pressure
        reference_pressure (float): Reference static pressure (usually freestream)
        dynamic_pressure (float): Dynamic pressure (0.5 * rho * V^2)
    
    Returns:
        float or array: Pressure coefficient (Cp)
    """
    # Ensure inputs are numpy arrays for element-wise operations
    pressure = np.asarray(pressure)
    
    # Calculate pressure coefficient
    with np.errstate(invalid='ignore', divide='ignore'):
        cp = (pressure - reference_pressure) / dynamic_pressure
    
    # Replace NaN and inf values with zeros
    cp = np.nan_to_num(cp)
    
    return cp


def calculate_dynamic_pressure(density, velocity):
    """Calculate dynamic pressure from density and velocity.
    
    The dynamic pressure is defined as:
    q = 0.5 * rho * V^2
    
    For F1 applications, this is crucial for determining aerodynamic forces
    at different speeds.
    
    Args:
        density (float or array): Fluid density
        velocity (float or array): Fluid velocity magnitude
    
    Returns:
        float or array: Dynamic pressure
    """
    # Ensure inputs are numpy arrays for element-wise operations
    density = np.asarray(density)
    velocity = np.asarray(velocity)
    
    # Calculate dynamic pressure
    dynamic_pressure = 0.5 * density * velocity**2
    
    return dynamic_pressure


def calculate_pressure_gradient(x, pressure):
    """Calculate pressure gradient along a coordinate.
    
    The pressure gradient is defined as:
    dp/dx = d(pressure)/d(x)
    
    This is useful for identifying adverse pressure gradients that
    can lead to flow separation - a critical factor in F1 aerodynamics.
    
    Args:
        x (array): Coordinate values
        pressure (array): Pressure values
    
    Returns:
        array: Pressure gradient
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    pressure = np.asarray(pressure)
    
    # Calculate pressure gradient using central differences
    gradient = np.zeros_like(pressure)
    
    # Forward difference for first point
    gradient[0] = (pressure[1] - pressure[0]) / (x[1] - x[0])
    
    # Central differences for interior points
    for i in range(1, len(x) - 1):
        gradient[i] = (pressure[i+1] - pressure[i-1]) / (x[i+1] - x[i-1])
    
    # Backward difference for last point
    gradient[-1] = (pressure[-1] - pressure[-2]) / (x[-1] - x[-2])
    
    return gradient


def calculate_pressure_forces(x, y, pressure, reference_pressure=0):
    """Calculate pressure forces on a closed contour.
    
    This function is particularly useful for calculating forces on
    aerodynamic surfaces like wings and diffusers in F1 cars.
    
    Args:
        x (array): x-coordinates of the contour
        y (array): y-coordinates of the contour
        pressure (array): Pressure values at each point
        reference_pressure (float, optional): Reference pressure to subtract
    
    Returns:
        tuple: (Fx, Fy) pressure forces in x and y directions
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    pressure = np.asarray(pressure)
    
    # Calculate pressure difference
    delta_p = pressure - reference_pressure
    
    # Calculate normal vectors for each segment
    n_x = np.zeros_like(x)
    n_y = np.zeros_like(y)
    
    # For a closed contour, connect the last point to the first
    x_extended = np.append(x, x[0])
    y_extended = np.append(y, y[0])
    
    # Calculate normal vectors (perpendicular to the contour)
    for i in range(len(x)):
        # Calculate tangent vector
        dx = x_extended[i+1] - x_extended[i]
        dy = y_extended[i+1] - y_extended[i]
        
        # Normal vector is perpendicular to tangent (clockwise rotation)
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            n_x[i] = dy / length
            n_y[i] = -dx / length
    
    # Calculate forces by integrating pressure * normal vector * segment length
    Fx = 0
    Fy = 0
    
    for i in range(len(x)):
        # Calculate segment length
        i_next = (i + 1) % len(x)
        dx = x[i_next] - x[i]
        dy = y[i_next] - y[i]
        segment_length = np.sqrt(dx**2 + dy**2)
        
        # Average pressure on the segment
        p_avg = (delta_p[i] + delta_p[i_next]) / 2
        
        # Add contribution to forces
        Fx += p_avg * n_x[i] * segment_length
        Fy += p_avg * n_y[i] * segment_length
    
    return Fx, Fy


def calculate_lift_drag_coefficients(Fx, Fy, dynamic_pressure, reference_area):
    """Calculate lift and drag coefficients from forces.
    
    For F1 applications, these are typically referred to as downforce and drag,
    with the lift coefficient being negative to indicate downward force.
    
    Args:
        Fx (float): Force in x-direction
        Fy (float): Force in y-direction
        dynamic_pressure (float): Dynamic pressure (0.5 * rho * V^2)
        reference_area (float): Reference area
    
    Returns:
        tuple: (Cl, Cd) lift and drag coefficients
    """
    # Calculate coefficients
    with np.errstate(invalid='ignore', divide='ignore'):
        Cd = Fx / (dynamic_pressure * reference_area)
        Cl = Fy / (dynamic_pressure * reference_area)
    
    # Replace NaN and inf values with zeros
    Cd = np.nan_to_num(Cd)
    Cl = np.nan_to_num(Cl)
    
    return Cl, Cd
