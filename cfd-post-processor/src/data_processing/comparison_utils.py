"""
Comparison Utilities Module

This module contains functions for comparing simulation results with experimental
or analytical data, and calculating error metrics.
"""

import numpy as np
from scipy.interpolate import interp1d


def compare_with_experimental(simulation_data, experimental_data, parameter, interpolate=True):
    """Compare simulation results with experimental data for a specific parameter.
    
    Args:
        simulation_data (dict): Dictionary containing simulation data
        experimental_data (dict): Dictionary containing experimental data
        parameter (str): Parameter to compare (e.g., 'pressure_coefficient', 'y_plus')
        interpolate (bool, optional): Whether to interpolate data to common coordinates
    
    Returns:
        dict: Dictionary containing comparison results
    """
    # Check if parameter exists in both datasets
    if parameter not in simulation_data or parameter not in experimental_data:
        raise ValueError(f"Parameter '{parameter}' not found in one or both datasets")
    
    # Extract data
    sim_data = simulation_data[parameter]
    exp_data = experimental_data[parameter]
    
    # Determine common coordinates
    if parameter == 'pressure_coefficient':
        sim_x = simulation_data['coordinates']['x']
        exp_x = experimental_data['coordinates']['x']
        
        if interpolate:
            # Interpolate simulation data to experimental coordinates
            interp_func = interp1d(sim_x, sim_data, kind='linear', bounds_error=False, fill_value='extrapolate')
            sim_data_interp = interp_func(exp_x)
            
            # Calculate differences
            diff = sim_data_interp - exp_data
            
            # Create comparison results
            comparison = {
                'coordinates': exp_x,
                'simulation_data': sim_data_interp,
                'experimental_data': exp_data,
                'difference': diff,
                'metrics': calculate_error_metrics(sim_data_interp, exp_data)
            }
        else:
            # Return data without interpolation
            comparison = {
                'simulation_coordinates': sim_x,
                'simulation_data': sim_data,
                'experimental_coordinates': exp_x,
                'experimental_data': exp_data
            }
    
    elif parameter == 'boundary_layer':
        sim_y_plus = simulation_data['boundary_layer']['y_plus']
        sim_u_plus = simulation_data['boundary_layer']['u_plus']
        exp_y_plus = experimental_data['boundary_layer']['y_plus']
        exp_u_plus = experimental_data['boundary_layer']['u_plus']
        
        if interpolate:
            # Interpolate simulation data to experimental y+ values
            # Use logarithmic interpolation for boundary layer data
            interp_func = interp1d(np.log(sim_y_plus), sim_u_plus, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            sim_u_plus_interp = interp_func(np.log(exp_y_plus))
            
            # Calculate differences
            diff = sim_u_plus_interp - exp_u_plus
            
            # Create comparison results
            comparison = {
                'y_plus': exp_y_plus,
                'simulation_u_plus': sim_u_plus_interp,
                'experimental_u_plus': exp_u_plus,
                'difference': diff,
                'metrics': calculate_error_metrics(sim_u_plus_interp, exp_u_plus)
            }
        else:
            # Return data without interpolation
            comparison = {
                'simulation_y_plus': sim_y_plus,
                'simulation_u_plus': sim_u_plus,
                'experimental_y_plus': exp_y_plus,
                'experimental_u_plus': exp_u_plus
            }
    
    else:
        # Generic comparison for other parameters
        comparison = {
            'simulation_data': sim_data,
            'experimental_data': exp_data
        }
    
    return comparison


def calculate_error_metrics(predicted, actual):
    """Calculate error metrics between predicted and actual values.
    
    Args:
        predicted (array): Predicted values (simulation)
        actual (array): Actual values (experimental or analytical)
    
    Returns:
        dict: Dictionary containing error metrics
    """
    # Ensure inputs are numpy arrays
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    
    # Remove NaN values
    mask = ~np.isnan(predicted) & ~np.isnan(actual)
    predicted = predicted[mask]
    actual = actual[mask]
    
    # Check if there are valid data points
    if len(predicted) == 0 or len(actual) == 0:
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }
    
    # Calculate error metrics
    error = predicted - actual
    abs_error = np.abs(error)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(abs_error)
    
    # Mean Squared Error (MSE)
    mse = np.mean(error**2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        mape = np.mean(np.abs(error / actual)) * 100
    
    # Replace NaN with zero
    mape = np.nan_to_num(mape)
    
    # Coefficient of Determination (R^2)
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum(error**2)
    
    if ss_total == 0:
        r2 = 0  # Avoid division by zero
    else:
        r2 = 1 - (ss_residual / ss_total)
    
    # Return metrics
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def compare_turbulence_models(model_results, parameter, reference_model=None):
    """Compare results from different turbulence models.
    
    Args:
        model_results (dict): Dictionary containing results from different models
                             {model_name: model_data, ...}
        parameter (str): Parameter to compare
        reference_model (str, optional): Name of the reference model for comparison
    
    Returns:
        dict: Dictionary containing comparison results
    """
    # Check if there are at least two models to compare
    if len(model_results) < 2:
        raise ValueError("At least two models are required for comparison")
    
    # If reference model is not specified, use the first model
    if reference_model is None:
        reference_model = list(model_results.keys())[0]
    
    # Check if reference model exists
    if reference_model not in model_results:
        raise ValueError(f"Reference model '{reference_model}' not found in model_results")
    
    # Extract reference model data
    ref_data = model_results[reference_model]
    
    # Check if parameter exists in reference model
    if parameter not in ref_data:
        raise ValueError(f"Parameter '{parameter}' not found in reference model")
    
    # Initialize comparison results
    comparison = {
        'reference_model': reference_model,
        'models': {},
        'differences': {},
        'metrics': {}
    }
    
    # Extract reference parameter data
    ref_param_data = ref_data[parameter]
    
    # Compare each model with the reference model
    for model_name, model_data in model_results.items():
        if model_name == reference_model:
            continue
        
        # Check if parameter exists in current model
        if parameter not in model_data:
            print(f"Warning: Parameter '{parameter}' not found in model '{model_name}'")
            continue
        
        # Extract parameter data for current model
        model_param_data = model_data[parameter]
        
        # Store model data
        comparison['models'][model_name] = model_param_data
        
        # Calculate difference
        if isinstance(ref_param_data, dict) and isinstance(model_param_data, dict):
            # Handle nested dictionaries (e.g., boundary_layer with y_plus and u_plus)
            comparison['differences'][model_name] = {}
            comparison['metrics'][model_name] = {}
            
            for key in ref_param_data:
                if key in model_param_data:
                    comparison['differences'][model_name][key] = model_param_data[key] - ref_param_data[key]
                    comparison['metrics'][model_name][key] = calculate_error_metrics(
                        model_param_data[key], ref_param_data[key]
                    )
        else:
            # Handle simple arrays
            comparison['differences'][model_name] = model_param_data - ref_param_data
            comparison['metrics'][model_name] = calculate_error_metrics(
                model_param_data, ref_param_data
            )
    
    return comparison


def compare_with_analytical(simulation_data, analytical_function, parameter, **kwargs):
    """Compare simulation results with an analytical function.
    
    Args:
        simulation_data (dict): Dictionary containing simulation data
        analytical_function (callable): Function that calculates analytical values
        parameter (str): Parameter to compare
        **kwargs: Additional arguments to pass to the analytical function
    
    Returns:
        dict: Dictionary containing comparison results
    """
    # Check if parameter exists in simulation data
    if parameter not in simulation_data:
        raise ValueError(f"Parameter '{parameter}' not found in simulation data")
    
    # Extract simulation data
    sim_data = simulation_data[parameter]
    
    # Determine input for analytical function based on parameter
    if parameter == 'boundary_layer':
        # For boundary layer, use y+ as input to calculate analytical u+
        y_plus = simulation_data['boundary_layer']['y_plus']
        sim_u_plus = simulation_data['boundary_layer']['u_plus']
        
        # Calculate analytical u+ values
        analytical_u_plus = analytical_function(y_plus, **kwargs)
        
        # Calculate differences
        diff = sim_u_plus - analytical_u_plus
        
        # Create comparison results
        comparison = {
            'y_plus': y_plus,
            'simulation_u_plus': sim_u_plus,
            'analytical_u_plus': analytical_u_plus,
            'difference': diff,
            'metrics': calculate_error_metrics(sim_u_plus, analytical_u_plus)
        }
    
    elif parameter == 'pressure_coefficient':
        # For pressure coefficient, use x-coordinate as input
        x = simulation_data['coordinates']['x']
        
        # Calculate analytical pressure coefficient
        analytical_cp = analytical_function(x, **kwargs)
        
        # Calculate differences
        diff = sim_data - analytical_cp
        
        # Create comparison results
        comparison = {
            'coordinates': x,
            'simulation_data': sim_data,
            'analytical_data': analytical_cp,
            'difference': diff,
            'metrics': calculate_error_metrics(sim_data, analytical_cp)
        }
    
    else:
        # Generic comparison for other parameters
        # Assume the analytical function knows how to handle the input
        analytical_data = analytical_function(sim_data, **kwargs)
        
        # Calculate differences
        diff = sim_data - analytical_data
        
        # Create comparison results
        comparison = {
            'simulation_data': sim_data,
            'analytical_data': analytical_data,
            'difference': diff,
            'metrics': calculate_error_metrics(sim_data, analytical_data)
        }
    
    return comparison
