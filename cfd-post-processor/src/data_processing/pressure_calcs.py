"""
Pressure calculations for CFD. Handles pressure coefficients, forces, and gradients.
"""

import numpy as np


def calculate_pressure_coefficient(pressure, reference_pressure, dynamic_pressure):
    """Figures out the pressure coefficient (Cp)."""
    pressure = np.asarray(pressure)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        cp = (pressure - reference_pressure) / dynamic_pressure
    
    cp = np.nan_to_num(cp)
    return cp


def calculate_dynamic_pressure(density, velocity):
    """Figures out the dynamic pressure (q = 0.5 * rho * V^2)."""
    density = np.asarray(density)
    velocity = np.asarray(velocity)
    dynamic_pressure = 0.5 * density * velocity**2
    return dynamic_pressure


def calculate_pressure_gradient(x, pressure):
    """Figures out how pressure changes along a coordinate."""
    x = np.asarray(x)
    pressure = np.asarray(pressure)
    
    gradient = np.zeros_like(pressure)
    
    gradient[0] = (pressure[1] - pressure[0]) / (x[1] - x[0])
    
    for i in range(1, len(x) - 1):
        gradient[i] = (pressure[i+1] - pressure[i-1]) / (x[i+1] - x[i-1])
    
    gradient[-1] = (pressure[-1] - pressure[-2]) / (x[-1] - x[-2])
    
    return gradient


def calculate_pressure_forces(x, y, pressure, reference_pressure=0):
    """Figures out the pressure forces on a closed contour."""
    x = np.asarray(x)
    y = np.asarray(y)
    pressure = np.asarray(pressure)
    
    delta_p = pressure - reference_pressure
    
    n_x = np.zeros_like(x)
    n_y = np.zeros_like(y)
    
    x_extended = np.append(x, x[0])
    y_extended = np.append(y, y[0])
    
    for i in range(len(x)):
        dx = x_extended[i+1] - x_extended[i]
        dy = y_extended[i+1] - y_extended[i]
        
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            n_x[i] = dy / length
            n_y[i] = -dx / length
    
    Fx = 0
    Fy = 0
    
    for i in range(len(x)):
        i_next = (i + 1) % len(x)
        dx = x[i_next] - x[i]
        dy = y[i_next] - y[i]
        segment_length = np.sqrt(dx**2 + dy**2)
        
        p_avg = (delta_p[i] + delta_p[i_next]) / 2
        
        Fx += p_avg * n_x[i] * segment_length
        Fy += p_avg * n_y[i] * segment_length
    
    return Fx, Fy


def calculate_lift_drag_coefficients(Fx, Fy, dynamic_pressure, reference_area):
    """Figures out the lift and drag coefficients from forces."""
    with np.errstate(invalid='ignore', divide='ignore'):
        Cd = Fx / (dynamic_pressure * reference_area)
        Cl = Fy / (dynamic_pressure * reference_area)
    
    Cd = np.nan_to_num(Cd)
    Cl = np.nan_to_num(Cl)
    
    return Cl, Cd
