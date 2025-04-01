"""
Turbulence Calculations Module

This module contains functions for calculating turbulence-related parameters
such as friction velocity (u_tau), dimensionless wall distance (y+),
and dimensionless velocity (u+). I've implemented these based on standard
wall treatment approaches used in CFD modeling.
"""

import numpy as np


def calculate_u_tau(density, wall_shear_stress):
    """Calculate friction velocity (u_tau) from density and wall shear stress.
    
    The friction velocity is defined as:
    u_tau = sqrt(tau_w / rho)
    
    where:
    - tau_w is the wall shear stress
    - rho is the fluid density
    
    Args:
        density (float or array): Fluid density
        wall_shear_stress (float or array): Wall shear stress
    
    Returns:
        float or array: Friction velocity (u_tau)
    """
    # Ensure inputs are numpy arrays for element-wise operations
    density = np.asarray(density)
    wall_shear_stress = np.asarray(wall_shear_stress)
    
    # Handle potential division by zero or negative values
    with np.errstate(invalid='ignore', divide='ignore'):
        u_tau = np.sqrt(np.abs(wall_shear_stress) / density)
    
    # Replace NaN values with zeros
    u_tau = np.nan_to_num(u_tau)
    
    return u_tau


def calculate_y_plus(distance, u_tau, density, viscosity):
    """Calculate y+ value from wall distance, friction velocity, density and viscosity.
    
    The dimensionless wall distance (y+) is defined as:
    y+ = (y * u_tau * rho) / mu
    
    This parameter is crucial for assessing mesh quality near walls and
    determining the appropriate wall treatment in turbulence models.
    
    Args:
        distance (float or array): Distance from the wall
        u_tau (float or array): Friction velocity
        density (float or array): Fluid density
        viscosity (float or array): Fluid dynamic viscosity
    
    Returns:
        float or array: Dimensionless wall distance (y+)
    """
    # Ensure inputs are numpy arrays for element-wise operations
    distance = np.asarray(distance)
    u_tau = np.asarray(u_tau)
    density = np.asarray(density)
    viscosity = np.asarray(viscosity)
    
    # Calculate y+
    with np.errstate(invalid='ignore', divide='ignore'):
        y_plus = distance * u_tau * density / viscosity
    
    # Replace NaN values with zeros
    y_plus = np.nan_to_num(y_plus)
    
    return y_plus


def calculate_u_plus(velocity, u_tau):
    """Calculate u+ value from velocity and friction velocity.
    
    The dimensionless velocity (u+) is defined as:
    u+ = u / u_tau
    
    This parameter is used to analyze boundary layer profiles and
    compare different flow regimes.
    
    Args:
        velocity (float or array): Velocity parallel to the wall
        u_tau (float or array): Friction velocity
    
    Returns:
        float or array: Dimensionless velocity (u+)
    """
    # Ensure inputs are numpy arrays for element-wise operations
    velocity = np.asarray(velocity)
    u_tau = np.asarray(u_tau)
    
    # Calculate u+
    with np.errstate(invalid='ignore', divide='ignore'):
        u_plus = velocity / u_tau
    
    # Replace NaN and inf values with zeros
    u_plus = np.nan_to_num(u_plus)
    
    return u_plus


def calculate_law_of_wall(y_plus):
    """Calculate the analytical law of the wall for comparison.
    
    The law of the wall consists of three regions:
    1. Viscous sublayer (y+ < 5): u+ = y+
    2. Buffer layer (5 <= y+ <= 30): Blend of viscous sublayer and log law
    3. Log-law region (y+ > 30): u+ = (1/kappa) * ln(y+) + B
    
    I've found this function particularly useful when validating
    turbulence models against theoretical profiles.
    
    Args:
        y_plus (array): Array of y+ values
    
    Returns:
        array: Corresponding u+ values according to the law of the wall
    """
    # Ensure input is a numpy array
    y_plus = np.asarray(y_plus)
    
    # Constants
    kappa = 0.41  # von Karman constant
    B = 5.5       # Log-law constant
    
    # Initialize u+ array
    u_plus = np.zeros_like(y_plus)
    
    # Viscous sublayer (y+ < 5)
    mask_viscous = y_plus < 5
    u_plus[mask_viscous] = y_plus[mask_viscous]
    
    # Log-law region (y+ > 30)
    mask_log = y_plus > 30
    u_plus[mask_log] = (1/kappa) * np.log(y_plus[mask_log]) + B
    
    # Buffer layer (5 <= y+ <= 30)
    # Use a blending function for smooth transition
    mask_buffer = (y_plus >= 5) & (y_plus <= 30)
    alpha = (y_plus[mask_buffer] - 5) / 25  # Blending factor (0 at y+=5, 1 at y+=30)
    u_plus_viscous = y_plus[mask_buffer]
    u_plus_log = (1/kappa) * np.log(y_plus[mask_buffer]) + B
    u_plus[mask_buffer] = (1 - alpha) * u_plus_viscous + alpha * u_plus_log
    
    return u_plus


def calculate_spalding_law(y_plus):
    """Calculate Spalding's law of the wall for comparison.
    
    Spalding's law provides a single formula valid for all regions of the boundary layer:
    y+ = u+ + exp(-kappa*B) * (exp(kappa*u+) - 1 - kappa*u+ - (kappa*u+)^2/2 - (kappa*u+)^3/6)
    
    This function numerically inverts this relationship to find u+ for given y+ values.
    I've included this as it's often more accurate than the piecewise law of the wall
    in the buffer region.
    
    Args:
        y_plus (array): Array of y+ values
    
    Returns:
        array: Corresponding u+ values according to Spalding's law
    """
    # Ensure input is a numpy array
    y_plus = np.asarray(y_plus)
    
    # Constants
    kappa = 0.41  # von Karman constant
    B = 5.5       # Log-law constant
    
    # Initialize u+ array with a reasonable guess
    u_plus = np.zeros_like(y_plus)
    
    # For y+ < 5, u+ ≈ y+
    mask_small = y_plus < 5
    u_plus[mask_small] = y_plus[mask_small]
    
    # For y+ > 30, u+ ≈ (1/kappa) * ln(y+) + B
    mask_large = y_plus > 30
    u_plus[mask_large] = (1/kappa) * np.log(y_plus[mask_large]) + B
    
    # For intermediate values, use numerical iteration
    mask_mid = ~(mask_small | mask_large)
    y_plus_mid = y_plus[mask_mid]
    
    # Initial guess for intermediate values
    u_plus_mid = np.sqrt(y_plus_mid)  # A reasonable initial guess
    
    # Newton-Raphson iteration to find u+ for given y+
    max_iter = 20
    tolerance = 1e-6
    
    for _ in range(max_iter):
        # Spalding's function: f(u+) = y+ - u+ - exp(-kappa*B) * (exp(kappa*u+) - 1 - kappa*u+ - (kappa*u+)^2/2 - (kappa*u+)^3/6)
        exp_kb = np.exp(-kappa * B)
        exp_ku = np.exp(kappa * u_plus_mid)
        ku = kappa * u_plus_mid
        ku2 = ku * ku
        ku3 = ku2 * ku
        
        f = y_plus_mid - u_plus_mid - exp_kb * (exp_ku - 1 - ku - ku2/2 - ku3/6)
        
        # Derivative of Spalding's function: f'(u+) = -1 - exp(-kappa*B) * (kappa*exp(kappa*u+) - kappa - kappa^2*u+ - (kappa^3*u+^2)/2)
        df = -1 - exp_kb * (kappa * exp_ku - kappa - kappa**2 * u_plus_mid - (kappa**3 * u_plus_mid**2)/2)
        
        # Update u+ using Newton-Raphson
        delta = f / df
        u_plus_mid = u_plus_mid - delta
        
        # Check for convergence
        if np.all(np.abs(delta) < tolerance):
            break
    
    # Update the result array
    u_plus[mask_mid] = u_plus_mid
    
    return u_plus
