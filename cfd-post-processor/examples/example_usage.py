"""
Example script for using the CFD post-processor

This script demonstrates how to use the CFD post-processor to analyze
airfoil simulation results from different turbulence models and compare them
with experimental data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import modules from the package
from src.data_extraction.file_readers import FLUENTFileReader, ExperimentalDataReader
from src.data_extraction.data_extractors import DataExtractor
from src.data_processing.turbulence_calcs import calculate_u_tau, calculate_y_plus, calculate_u_plus
from src.data_processing.pressure_calcs import calculate_pressure_coefficient, calculate_dynamic_pressure
from src.data_processing.comparison_utils import compare_with_experimental, compare_turbulence_models
from src.visualization.pressure_plots import plot_pressure_coefficient, plot_pressure_comparison
from src.visualization.boundary_layer_plots import plot_boundary_layer_profile, plot_y_plus_distribution
from src.visualization.convergence_plots import plot_convergence_history
from src.visualization.contour_plots import plot_contour, plot_vector_field
from src.visualization.plot_utils import set_plot_style, save_figure


def main():
    """Main function to demonstrate the CFD post-processor functionality."""
    print("CFD Post-Processor Example - Airfoil Analysis")
    
    # Create output directory
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    set_plot_style(style='scientific', font_size=12)
    
    # Example 1: Load and process simulation data
    print("\nExample 1: Loading and processing airfoil simulation data")
    
    # In a real scenario, you would load actual simulation files
    # For this example, we'll generate sample data for a NACA 0012 airfoil
    
    # Generate sample airfoil coordinates (NACA 0012)
    x = np.linspace(0, 1, 100)
    y_upper = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    y_lower = -y_upper
    
    # Generate sample pressure data for different turbulence models
    cp_model1 = -0.5 * (1 - 4 * (x - 0.5)**2) + 0.1 * np.random.randn(len(x))  # Standard wall function
    cp_model2 = -0.55 * (1 - 4.2 * (x - 0.5)**2) + 0.05 * np.random.randn(len(x))  # Enhanced wall function
    cp_model3 = -0.52 * (1 - 4.1 * (x - 0.5)**2) + 0.08 * np.random.randn(len(x))  # Reynolds Stress Model
    
    # Generate sample experimental data
    x_exp = np.linspace(0, 1, 20)
    cp_exp = -0.53 * (1 - 4.1 * (x_exp - 0.5)**2) + 0.1 * np.random.randn(len(x_exp))
    
    # Example 2: Pressure coefficient analysis
    print("\nExample 2: Airfoil pressure coefficient analysis")
    
    # Plot pressure coefficient for all models
    fig, ax = plot_pressure_coefficient(
        [x, x, x],
        [cp_model1, cp_model2, cp_model3],
        labels=['Standard Wall Function', 'Enhanced Wall Function', 'Reynolds Stress Model'],
        title="Airfoil Pressure Coefficient Distribution",
        experimental_data=(x_exp, cp_exp)
    )
    
    # Save the figure
    save_figure(fig, os.path.join(output_dir, 'pressure_coefficient'), formats=['png'])
    plt.close(fig)
    
    # Example 3: Boundary layer analysis
    print("\nExample 3: Airfoil boundary layer analysis")
    
    # Generate sample boundary layer data
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
    
    # Plot boundary layer profile for all models
    fig, ax = plot_boundary_layer_profile(
        [y_plus_values, y_plus_values, y_plus_values],
        [u_plus_model1, u_plus_model2, u_plus_model3],
        labels=['Standard Wall Function', 'Enhanced Wall Function', 'Reynolds Stress Model'],
        title="Airfoil Boundary Layer Profile",
        show_analytical=True
    )
    
    # Save the figure
    save_figure(fig, os.path.join(output_dir, 'boundary_layer_profile'), formats=['png'])
    plt.close(fig)
    
    # Example 4: Convergence history analysis
    print("\nExample 4: Convergence history analysis")
    
    # Generate sample convergence history data
    iterations = np.arange(1, 601)
    
    # Model 1: Standard wall function (converges faster)
    residuals_model1 = np.zeros((5, len(iterations)))
    residuals_model1[0] = 1e-1 * np.exp(-iterations / 100) + 1e-6 * np.random.randn(len(iterations))  # Continuity
    residuals_model1[1] = 5e-2 * np.exp(-iterations / 80) + 1e-6 * np.random.randn(len(iterations))   # X-velocity
    residuals_model1[2] = 5e-2 * np.exp(-iterations / 90) + 1e-6 * np.random.randn(len(iterations))   # Y-velocity
    residuals_model1[3] = 2e-2 * np.exp(-iterations / 70) + 1e-6 * np.random.randn(len(iterations))   # k
    residuals_model1[4] = 3e-2 * np.exp(-iterations / 85) + 1e-6 * np.random.randn(len(iterations))   # epsilon
    
    # Plot convergence history for model 1
    fig, ax = plot_convergence_history(
        iterations,
        residuals_model1,
        parameter_names=['Continuity', 'X-Velocity', 'Y-Velocity', 'k', 'epsilon'],
        title="Convergence History - Standard Wall Function",
        convergence_threshold=1e-4
    )
    
    # Save the figure
    save_figure(fig, os.path.join(output_dir, 'convergence_history'), formats=['png'])
    plt.close(fig)
    
    # Example 5: Flow field visualization
    print("\nExample 5: Flow field visualization around airfoil")
    
    # Generate sample 2D data for contour plot
    # Create a simple airfoil shape for visualization
    theta = np.linspace(0, 2*np.pi, 100)
    airfoil_x = 0.5 + 0.3 * np.cos(theta) + 0.1 * np.cos(2*theta)
    airfoil_y = 0.1 * np.sin(theta)
    
    # Create a grid around the airfoil
    x_grid = np.linspace(0, 1, 100)
    y_grid = np.linspace(-0.3, 0.3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate distance from each point to the airfoil
    R = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Find minimum distance to airfoil
            distances = np.sqrt((X[i,j] - airfoil_x)**2 + (Y[i,j] - airfoil_y)**2)
            R[i,j] = np.min(distances)
    
    # Create a mask for points inside the airfoil
    mask = R < 0.05
    
    # Generate sample pressure field
    pressure = np.zeros_like(X)
    pressure[~mask] = 1 - 2 * np.exp(-5 * (X[~mask] - 0.5)**2) * np.exp(-10 * Y[~mask]**2)
    
    # Generate sample velocity field
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    # Simple flow field around airfoil
    u[~mask] = 1.0 - 0.5 * np.exp(-10 * (Y[~mask])**2) * np.sin(2 * np.pi * X[~mask])
    v[~mask] = 0.2 * np.exp(-10 * (Y[~mask])**2) * np.cos(2 * np.pi * X[~mask])
    
    # Plot pressure contour
    fig, ax = plot_contour(
        X, Y, pressure,
        title="Pressure Distribution Around Airfoil",
        colorbar_label="Pressure",
        levels=20
    )
    
    # Add airfoil outline
    ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
    
    # Save the figure
    save_figure(fig, os.path.join(output_dir, 'pressure_contour'), formats=['png'])
    plt.close(fig)
    
    # Plot velocity vector field
    fig, ax = plot_vector_field(
        X, Y, u, v,
        title="Velocity Field Around Airfoil",
        density=5,
        add_magnitude=True
    )
    
    # Add airfoil outline
    ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
    
    # Save the figure
    save_figure(fig, os.path.join(output_dir, 'velocity_field'), formats=['png'])
    plt.close(fig)
    
    print(f"\nAll examples completed. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
