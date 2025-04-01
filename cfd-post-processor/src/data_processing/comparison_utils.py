"""
Functions for comparing CFD results with experimental and analytical data.
"""

import numpy as np
from scipy.interpolate import interp1d


def compare_with_experimental(simulation_data, experimental_data, parameter, interpolate=True):
    """Compares simulation results with experimental data."""
    if parameter not in simulation_data or parameter not in experimental_data:
        raise ValueError(f"Parameter '{parameter}' not found in one or both datasets")
    
    sim_data = simulation_data[parameter]
    exp_data = experimental_data[parameter]
    
    if parameter == 'pressure_coefficient':
        sim_x = simulation_data['coordinates']['x']
        exp_x = experimental_data['coordinates']['x']
        
        if interpolate:
            interp_func = interp1d(sim_x, sim_data, kind='linear', bounds_error=False, fill_value='extrapolate')
            sim_data_interp = interp_func(exp_x)
            
            diff = sim_data_interp - exp_data
            
            comparison = {
                'coordinates': exp_x,
                'simulation_data': sim_data_interp,
                'experimental_data': exp_data,
                'difference': diff,
                'metrics': calculate_error_metrics(sim_data_interp, exp_data)
            }
        else:
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
            interp_func = interp1d(np.log(sim_y_plus), sim_u_plus, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            sim_u_plus_interp = interp_func(np.log(exp_y_plus))
            
            diff = sim_u_plus_interp - exp_u_plus
            
            comparison = {
                'y_plus': exp_y_plus,
                'simulation_u_plus': sim_u_plus_interp,
                'experimental_u_plus': exp_u_plus,
                'difference': diff,
                'metrics': calculate_error_metrics(sim_u_plus_interp, exp_u_plus)
            }
        else:
            comparison = {
                'simulation_y_plus': sim_y_plus,
                'simulation_u_plus': sim_u_plus,
                'experimental_y_plus': exp_y_plus,
                'experimental_u_plus': exp_u_plus
            }
    
    else:
        comparison = {
            'simulation_data': sim_data,
            'experimental_data': exp_data
        }
    
    return comparison


def calculate_error_metrics(predicted, actual):
    """Figures out how well the predictions match the actual values."""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    
    mask = ~np.isnan(predicted) & ~np.isnan(actual)
    predicted = predicted[mask]
    actual = actual[mask]
    
    if len(predicted) == 0 or len(actual) == 0:
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }
    
    error = predicted - actual
    abs_error = np.abs(error)
    
    mae = np.mean(abs_error)
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        mape = np.mean(np.abs(error / actual)) * 100
    
    mape = np.nan_to_num(mape)
    
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum(error**2)
    
    if ss_total == 0:
        r2 = 0
    else:
        r2 = 1 - (ss_residual / ss_total)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def compare_turbulence_models(model_results, parameter, reference_model=None):
    """Compares results from different turbulence models."""
    if len(model_results) < 2:
        raise ValueError("Need at least two models to compare")
    
    if reference_model is None:
        reference_model = list(model_results.keys())[0]
    
    if reference_model not in model_results:
        raise ValueError(f"Reference model '{reference_model}' not found in model_results")
    
    ref_data = model_results[reference_model]
    
    if parameter not in ref_data:
        raise ValueError(f"Parameter '{parameter}' not found in reference model")
    
    comparison = {
        'reference_model': reference_model,
        'models': {},
        'differences': {},
        'metrics': {}
    }
    
    ref_param_data = ref_data[parameter]
    
    for model_name, model_data in model_results.items():
        if model_name == reference_model:
            continue
        
        if parameter not in model_data:
            print(f"Warning: Parameter '{parameter}' not found in model '{model_name}'")
            continue
        
        model_param_data = model_data[parameter]
        
        comparison['models'][model_name] = model_param_data
        
        if isinstance(ref_param_data, dict) and isinstance(model_param_data, dict):
            comparison['differences'][model_name] = {}
            comparison['metrics'][model_name] = {}
            
            for key in ref_param_data:
                if key in model_param_data:
                    comparison['differences'][model_name][key] = model_param_data[key] - ref_param_data[key]
                    comparison['metrics'][model_name][key] = calculate_error_metrics(
                        model_param_data[key], ref_param_data[key]
                    )
        else:
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
