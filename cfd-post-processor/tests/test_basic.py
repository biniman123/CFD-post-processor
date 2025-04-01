"""
Test script for CFD post-processor

This script tests the basic functionality of the CFD post-processor package
by creating sample data and running it through the processing pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Add the src directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from the package
from src.data_processing.turbulence_calcs import calculate_u_tau, calculate_y_plus, calculate_u_plus
from src.data_processing.pressure_calcs import calculate_pressure_coefficient, calculate_dynamic_pressure
from src.visualization.pressure_plots import plot_pressure_coefficient
from src.visualization.boundary_layer_plots import plot_boundary_layer_profile
from src.visualization.convergence_plots import plot_convergence_history


def generate_sample_data():
    """Generate sample CFD data for testing."""
    print("Generating sample CFD data...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate aerofoil coordinates (NACA 0012)
    x = np.linspace(0, 1, 100)
    y_upper = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    y_lower = -y_upper
    
    # Generate pressure coefficient data for different turbulence models
    cp_model1 = -0.5 * (1 - 4 * (x - 0.5)**2) + 0.1 * np.random.randn(len(x))  # Standard wall function
    cp_model2 = -0.55 * (1 - 4.2 * (x - 0.5)**2) + 0.05 * np.random.randn(len(x))  # Enhanced wall function
    cp_model3 = -0.52 * (1 - 4.1 * (x - 0.5)**2) + 0.08 * np.random.randn(len(x))  # Reynolds Stress Model
    
    # Generate experimental data (with some offset)
    x_exp = np.linspace(0, 1, 20)
    cp_exp = -0.53 * (1 - 4.1 * (x_exp - 0.5)**2) + 0.1 * np.random.randn(len(x_exp))
    
    # Generate boundary layer data
    y_plus_values = np.logspace(0, 3, 50)
    
    # Model 1: Standard wall function
    u_plus_model1 = np.zeros_like(y_plus_values)
    mask1 = y_plus_values < 5
    mask2 = (y_plus_values >= 5) & (y_plus_values <= 30)
    mask3 = y_plus_values > 30
    
    u_plus_model1[mask1] = y_plus_values[mask1]  # Viscous sublayer
    u_plus_model1[mask3] = (1/0.41) * np.log(y_plus_values[mask3]) + 5.5  # Log-law region
    
    # Blend for buffer region
    alpha = (y_plus_values[mask2] - 5) / 25
    u_plus_viscous = y_plus_values[mask2]
    u_plus_log = (1/0.41) * np.log(y_plus_values[mask2]) + 5.5
    u_plus_model1[mask2] = (1 - alpha) * u_plus_viscous + alpha * u_plus_log
    
    # Add some random noise
    u_plus_model1 += 0.5 * np.random.randn(len(y_plus_values))
    
    # Model 2: Enhanced wall function (closer to analytical)
    u_plus_model2 = np.zeros_like(y_plus_values)
    u_plus_model2[mask1] = y_plus_values[mask1]  # Viscous sublayer
    u_plus_model2[mask3] = (1/0.41) * np.log(y_plus_values[mask3]) + 5.5  # Log-law region
    
    # Blend for buffer region
    u_plus_model2[mask2] = (1 - alpha) * u_plus_viscous + alpha * u_plus_log
    
    # Add less random noise
    u_plus_model2 += 0.2 * np.random.randn(len(y_plus_values))
    
    # Model 3: Reynolds Stress Model (slightly different)
    u_plus_model3 = np.zeros_like(y_plus_values)
    u_plus_model3[mask1] = y_plus_values[mask1] * 1.05  # Slightly different viscous sublayer
    u_plus_model3[mask3] = (1/0.41) * np.log(y_plus_values[mask3]) + 5.3  # Slightly different log-law
    
    # Blend for buffer region
    u_plus_model3[mask2] = (1 - alpha) * (y_plus_values[mask2] * 1.05) + alpha * ((1/0.41) * np.log(y_plus_values[mask2]) + 5.3)
    
    # Add some random noise
    u_plus_model3 += 0.3 * np.random.randn(len(y_plus_values))
    
    # Generate convergence history data
    iterations = np.arange(1, 601)
    
    # Model 1: Standard wall function (converges faster)
    residuals_model1 = np.zeros((5, len(iterations)))
    residuals_model1[0] = 1e-1 * np.exp(-iterations / 100) + 1e-6 * np.random.randn(len(iterations))  # Continuity
    residuals_model1[1] = 5e-2 * np.exp(-iterations / 80) + 1e-6 * np.random.randn(len(iterations))   # X-velocity
    residuals_model1[2] = 5e-2 * np.exp(-iterations / 90) + 1e-6 * np.random.randn(len(iterations))   # Y-velocity
    residuals_model1[3] = 2e-2 * np.exp(-iterations / 70) + 1e-6 * np.random.randn(len(iterations))   # k
    residuals_model1[4] = 3e-2 * np.exp(-iterations / 85) + 1e-6 * np.random.randn(len(iterations))   # epsilon
    
    # Model 2: Enhanced wall function (similar convergence)
    residuals_model2 = np.zeros((5, len(iterations)))
    residuals_model2[0] = 1.2e-1 * np.exp(-iterations / 110) + 1e-6 * np.random.randn(len(iterations))  # Continuity
    residuals_model2[1] = 6e-2 * np.exp(-iterations / 85) + 1e-6 * np.random.randn(len(iterations))     # X-velocity
    residuals_model2[2] = 6e-2 * np.exp(-iterations / 95) + 1e-6 * np.random.randn(len(iterations))     # Y-velocity
    residuals_model2[3] = 2.5e-2 * np.exp(-iterations / 75) + 1e-6 * np.random.randn(len(iterations))   # k
    residuals_model2[4] = 3.5e-2 * np.exp(-iterations / 90) + 1e-6 * np.random.randn(len(iterations))   # epsilon
    
    # Model 3: Reynolds Stress Model (slower convergence)
    iterations_model3 = np.arange(1, 3001)
    residuals_model3 = np.zeros((8, len(iterations_model3)))
    residuals_model3[0] = 2e-1 * np.exp(-iterations_model3 / 500) + 1e-6 * np.random.randn(len(iterations_model3))  # Continuity
    residuals_model3[1] = 1e-1 * np.exp(-iterations_model3 / 450) + 1e-6 * np.random.randn(len(iterations_model3))  # X-velocity
    residuals_model3[2] = 1e-1 * np.exp(-iterations_model3 / 470) + 1e-6 * np.random.randn(len(iterations_model3))  # Y-velocity
    residuals_model3[3] = 5e-2 * np.exp(-iterations_model3 / 400) + 1e-6 * np.random.randn(len(iterations_model3))  # k
    residuals_model3[4] = 7e-2 * np.exp(-iterations_model3 / 420) + 1e-6 * np.random.randn(len(iterations_model3))  # epsilon
    residuals_model3[5] = 8e-2 * np.exp(-iterations_model3 / 430) + 1e-6 * np.random.randn(len(iterations_model3))  # uu-stress
    residuals_model3[6] = 8e-2 * np.exp(-iterations_model3 / 440) + 1e-6 * np.random.randn(len(iterations_model3))  # vv-stress
    residuals_model3[7] = 9e-2 * np.exp(-iterations_model3 / 450) + 1e-6 * np.random.randn(len(iterations_model3))  # uv-stress
    
    # Return the generated data
    return {
        'aerofoil': {
            'x': x,
            'y_upper': y_upper,
            'y_lower': y_lower
        },
        'pressure': {
            'x': x,
            'cp_model1': cp_model1,
            'cp_model2': cp_model2,
            'cp_model3': cp_model3,
            'x_exp': x_exp,
            'cp_exp': cp_exp
        },
        'boundary_layer': {
            'y_plus': y_plus_values,
            'u_plus_model1': u_plus_model1,
            'u_plus_model2': u_plus_model2,
            'u_plus_model3': u_plus_model3
        },
        'convergence': {
            'iterations_model1': iterations,
            'residuals_model1': residuals_model1,
            'iterations_model2': iterations,
            'residuals_model2': residuals_model2,
            'iterations_model3': iterations_model3,
            'residuals_model3': residuals_model3
        },
        'output_dir': output_dir
    }


def test_data_processing(data):
    """Test data processing functions."""
    print("\nTesting data processing functions...")
    
    # Test turbulence calculations
    print("Testing turbulence calculations...")
    
    # Sample data
    density = 1.225  # kg/m^3
    wall_shear_stress = 2.5  # Pa
    distance = 0.0001  # m
    viscosity = 1.8e-5  # kg/(m*s)
    velocity = 50.0  # m/s
    
    # Calculate u_tau
    u_tau = calculate_u_tau(density, wall_shear_stress)
    print(f"u_tau = {u_tau:.4f} m/s")
    
    # Calculate y+
    y_plus = calculate_y_plus(distance, u_tau, density, viscosity)
    print(f"y+ = {y_plus:.4f}")
    
    # Calculate u+
    u_plus = calculate_u_plus(velocity, u_tau)
    print(f"u+ = {u_plus:.4f}")
    
    # Test pressure calculations
    print("\nTesting pressure calculations...")
    
    # Sample data
    pressure = 101500.0  # Pa
    reference_pressure = 101325.0  # Pa
    velocity = 100.0  # m/s
    
    # Calculate dynamic pressure
    dynamic_pressure = calculate_dynamic_pressure(density, velocity)
    print(f"Dynamic pressure = {dynamic_pressure:.2f} Pa")
    
    # Calculate pressure coefficient
    cp = calculate_pressure_coefficient(pressure, reference_pressure, dynamic_pressure)
    print(f"Pressure coefficient = {cp:.4f}")
    
    return True


def test_visualization(data):
    """Test visualization functions."""
    print("\nTesting visualization functions...")
    
    # Create output directory
    output_dir = data['output_dir']
    
    # Test pressure coefficient plot
    print("Testing pressure coefficient plot...")
    x = data['pressure']['x']
    cp_model1 = data['pressure']['cp_model1']
    cp_model2 = data['pressure']['cp_model2']
    cp_model3 = data['pressure']['cp_model3']
    x_exp = data['pressure']['x_exp']
    cp_exp = data['pressure']['cp_exp']
    
    # Plot pressure coefficient for all models
    fig, ax = plot_pressure_coefficient(
        [x, x, x],
        [cp_model1, cp_model2, cp_model3],
        labels=['Standard Wall Function', 'Enhanced Wall Function', 'Reynolds Stress Model'],
        title="Pressure Coefficient Distribution",
        experimental_data=(x_exp, cp_exp),
        style='scientific'
    )
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'pressure_coefficient.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Test boundary layer profile plot
    print("Testing boundary layer profile plot...")
    y_plus = data['boundary_layer']['y_plus']
    u_plus_model1 = data['boundary_layer']['u_plus_model1']
    u_plus_model2 = data['boundary_layer']['u_plus_model2']
    u_plus_model3 = data['boundary_layer']['u_plus_model3']
    
    # Plot boundary layer profile for all models
    fig, ax = plot_boundary_layer_profile(
        [y_plus, y_plus, y_plus],
        [u_plus_model1, u_plus_model2, u_plus_model3],
        labels=['Standard Wall Function', 'Enhanced Wall Function', 'Reynolds Stress Model'],
        title="Boundary Layer Profile",
        style='scientific',
        show_analytical=True
    )
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'boundary_layer_profile.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Test convergence history plot
    print("Testing convergence history plot...")
    iterations_model1 = data['convergence']['iterations_model1']
    residuals_model1 = data['convergence']['residuals_model1']
    
    # Plot convergence history for model 1
    fig, ax = plot_convergence_history(
        iterations_model1,
        residuals_model1,
        parameter_names=['Continuity', 'X-Velocity', 'Y-Velocity', 'k', 'epsilon'],
        title="Convergence History - Standard Wall Function",
        style='scientific',
        convergence_threshold=1e-4
    )
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'convergence_history_model1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot convergence history for model 3 (Reynolds Stress Model)
    iterations_model3 = data['convergence']['iterations_model3']
    residuals_model3 = data['convergence']['residuals_model3']
    
    fig, ax = plot_convergence_history(
        iterations_model3,
        residuals_model3,
        parameter_names=['Continuity', 'X-Velocity', 'Y-Velocity', 'k', 'epsilon', 'uu-stress', 'vv-stress', 'uv-stress'],
        title="Convergence History - Reynolds Stress Model",
        style='scientific',
        convergence_threshold=1e-4
    )
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'convergence_history_model3.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization tests completed. Output saved to {output_dir}")
    return True


def main():
    """Main function to run all tests."""
    print("Starting CFD post-processor tests...")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Test data processing functions
    test_data_processing(data)
    
    # Test visualization functions
    test_visualization(data)
    
    print("\nAll tests completed successfully!")
    print(f"Test output saved to {data['output_dir']}")


if __name__ == "__main__":
    main()
