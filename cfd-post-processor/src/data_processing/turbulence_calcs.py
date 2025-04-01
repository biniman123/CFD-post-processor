"""
Functions for figuring out turbulence stuff in CFD.
"""

import numpy as np


def calculate_u_tau(density, wall_shear_stress):
    """Figures out the friction velocity from density and wall shear stress."""
    density = np.asarray(density)
    wall_shear_stress = np.asarray(wall_shear_stress)
    with np.errstate(invalid='ignore', divide='ignore'):
        u_tau = np.sqrt(np.abs(wall_shear_stress) / density)
    return np.nan_to_num(u_tau)


def calculate_y_plus(distance, u_tau, density, viscosity):
    """Figures out the dimensionless wall distance (y+)."""
    distance = np.asarray(distance)
    u_tau = np.asarray(u_tau)
    density = np.asarray(density)
    viscosity = np.asarray(viscosity)
    with np.errstate(invalid='ignore', divide='ignore'):
        y_plus = distance * u_tau * density / viscosity
    return np.nan_to_num(y_plus)


def calculate_u_plus(velocity, u_tau):
    """Figures out the dimensionless velocity (u+)."""
    velocity = np.asarray(velocity)
    u_tau = np.asarray(u_tau)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        u_plus = velocity / u_tau
    
    u_plus = np.nan_to_num(u_plus)
    return u_plus


def calculate_law_of_wall(y_plus):
    """Figures out the analytical law of the wall for comparison."""
    y_plus = np.asarray(y_plus)
    kappa = 0.41
    B = 5.5
    u_plus = np.zeros_like(y_plus)
    
    mask_viscous = y_plus < 5
    mask_log = y_plus > 30
    mask_buffer = (y_plus >= 5) & (y_plus <= 30)
    
    u_plus[mask_viscous] = y_plus[mask_viscous]
    u_plus[mask_log] = (1/kappa) * np.log(y_plus[mask_log]) + B
    
    alpha = (y_plus[mask_buffer] - 5) / 25
    u_plus_viscous = y_plus[mask_buffer]
    u_plus_log = (1/kappa) * np.log(y_plus[mask_buffer]) + B
    u_plus[mask_buffer] = (1 - alpha) * u_plus_viscous + alpha * u_plus_log
    
    return u_plus


def calculate_spalding_law(y_plus):
    """Figures out Spalding's law of the wall."""
    y_plus = np.asarray(y_plus)
    kappa = 0.41
    B = 5.5
    u_plus = np.zeros_like(y_plus)
    
    mask_small = y_plus < 5
    mask_large = y_plus > 30
    mask_mid = ~(mask_small | mask_large)
    
    u_plus[mask_small] = y_plus[mask_small]
    u_plus[mask_large] = (1/kappa) * np.log(y_plus[mask_large]) + B
    
    y_plus_mid = y_plus[mask_mid]
    u_plus_mid = np.sqrt(y_plus_mid)
    
    max_iter = 20
    tolerance = 1e-6
    
    for _ in range(max_iter):
        exp_kb = np.exp(-kappa * B)
        exp_ku = np.exp(kappa * u_plus_mid)
        ku = kappa * u_plus_mid
        ku2 = ku * ku
        ku3 = ku2 * ku
        
        f = y_plus_mid - u_plus_mid - exp_kb * (exp_ku - 1 - ku - ku2/2 - ku3/6)
        df = -1 - exp_kb * (kappa * exp_ku - kappa - kappa**2 * u_plus_mid - (kappa**3 * u_plus_mid**2)/2)
        
        delta = f / df
        u_plus_mid = u_plus_mid - delta
        
        if np.all(np.abs(delta) < tolerance):
            break
    
    u_plus[mask_mid] = u_plus_mid
    return u_plus
