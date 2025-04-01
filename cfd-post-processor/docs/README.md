# CFD Post-Processing Tool Documentation

This documentation provides detailed information on how to use the CFD post-processing tool for analyzing and visualizing Computational Fluid Dynamics (CFD) simulation results, with specific support for ANSYS Fluent and F1 aerodynamics applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Module Overview](#module-overview)
4. [Data Extraction](#data-extraction)
5. [Data Processing](#data-processing)
6. [Visualization](#visualization)
7. [ANSYS Fluent Integration](#ansys-fluent-integration)
8. [F1 Aerodynamics Applications](#f1-aerodynamics-applications)
9. [Examples](#examples)
10. [API Reference](#api-reference)

## Introduction

The CFD Post-Processing Tool helps engineers and researchers analyze simulation results from different turbulence models, extract key parameters, and create publication-quality visualizations. I developed this tool based on my experience with aerodynamic simulations and the need for flexible, customizable post-processing capabilities beyond what's available in commercial software.

The tool supports:

- Reading data from various CFD simulation output formats, with special focus on ANSYS Fluent
- Calculating important parameters like Y+, U+, and pressure coefficients
- Comparing results from different turbulence models
- Creating various types of visualizations for reports and presentations

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib
- Pandas

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cfd-post-processor.git
   cd cfd-post-processor
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib pandas
   ```

## Module Overview

The tool is organized into three main modules:

1. **Data Extraction**: Modules for reading and extracting data from simulation files
2. **Data Processing**: Modules for processing and analyzing simulation data
3. **Visualization**: Modules for creating visualizations of simulation results

## Data Extraction

### File Readers

The `file_readers.py` module provides classes for reading different types of CFD simulation files:

- `CFDFileReader`: Base class for all file readers
- `FLUENTFileReader`: Reader for ANSYS FLUENT output files
- `CSVFileReader`: Reader for generic CSV data files
- `ExperimentalDataReader`: Reader for experimental data files

Example usage:

```python
from src.data_extraction.file_readers import FLUENTFileReader

# Create a reader for a FLUENT output file
reader = FLUENTFileReader('path/to/fluent_output.csv')

# Read the data
data = reader.read()
```

### Data Extractors

The `data_extractors.py` module provides classes for extracting specific data from simulation results:

- `DataExtractor`: Extract data along lines, on surfaces, and from convergence history
- `CoordinateUtils`: Utilities for handling coordinate systems

Example usage:

```python
from src.data_extraction.data_extractors import DataExtractor

# Create a data extractor
extractor = DataExtractor(data)

# Extract data along a line
line_data = extractor.extract_line_data(
    start_point=(0, 0),
    end_point=(1, 0),
    num_points=100
)

# Extract data on the aerofoil surface
aerofoil_data = extractor.extract_aerofoil_data()
```

## Data Processing

### Turbulence Calculations

The `turbulence_calcs.py` module provides functions for calculating turbulence-related parameters:

- `calculate_u_tau`: Calculate friction velocity
- `calculate_y_plus`: Calculate dimensionless wall distance (y+)
- `calculate_u_plus`: Calculate dimensionless velocity (u+)
- `calculate_law_of_wall`: Calculate analytical law of the wall
- `calculate_spalding_law`: Calculate Spalding's law of the wall

Example usage:

```python
from src.data_processing.turbulence_calcs import calculate_y_plus, calculate_u_plus

# Calculate y+
y_plus = calculate_y_plus(
    distance=0.0001,  # m
    u_tau=1.2,        # m/s
    density=1.225,    # kg/m^3
    viscosity=1.8e-5  # kg/(m*s)
)

# Calculate u+
u_plus = calculate_u_plus(
    velocity=20.0,  # m/s
    u_tau=1.2       # m/s
)
```

### Pressure Calculations

The `pressure_calcs.py` module provides functions for calculating pressure-related parameters:

- `calculate_pressure_coefficient`: Calculate pressure coefficient
- `calculate_dynamic_pressure`: Calculate dynamic pressure
- `calculate_pressure_gradient`: Calculate pressure gradient
- `calculate_pressure_forces`: Calculate pressure forces on a closed contour
- `calculate_lift_drag_coefficients`: Calculate lift and drag coefficients

Example usage:

```python
from src.data_processing.pressure_calcs import calculate_pressure_coefficient, calculate_dynamic_pressure

# Calculate dynamic pressure
dynamic_pressure = calculate_dynamic_pressure(
    density=1.225,  # kg/m^3
    velocity=100.0  # m/s
)

# Calculate pressure coefficient
cp = calculate_pressure_coefficient(
    pressure=101500.0,          # Pa
    reference_pressure=101325.0, # Pa
    dynamic_pressure=dynamic_pressure
)
```

### Comparison Utilities

The `comparison_utils.py` module provides functions for comparing simulation results:

- `compare_with_experimental`: Compare simulation results with experimental data
- `calculate_error_metrics`: Calculate error metrics between predicted and actual values
- `compare_turbulence_models`: Compare results from different turbulence models
- `compare_with_analytical`: Compare simulation results with analytical functions

Example usage:

```python
from src.data_processing.comparison_utils import compare_with_experimental, calculate_error_metrics

# Compare simulation results with experimental data
comparison = compare_with_experimental(
    simulation_data=sim_data,
    experimental_data=exp_data,
    parameter='pressure_coefficient',
    interpolate=True
)

# Calculate error metrics
metrics = calculate_error_metrics(
    predicted=sim_values,
    actual=exp_values
)
```

## Visualization

### Pressure Plots

The `pressure_plots.py` module provides functions for plotting pressure-related data:

- `plot_pressure_coefficient`: Plot pressure coefficient distribution
- `plot_pressure_contour`: Plot pressure contour on a 2D domain
- `plot_pressure_comparison`: Plot pressure coefficient comparison for multiple turbulence models

Example usage:

```python
from src.visualization.pressure_plots import plot_pressure_coefficient

# Plot pressure coefficient distribution
fig, ax = plot_pressure_coefficient(
    x_coords=[x1, x2, x3],
    cp_values=[cp1, cp2, cp3],
    labels=['Model 1', 'Model 2', 'Model 3'],
    title="Pressure Coefficient Distribution",
    experimental_data=(x_exp, cp_exp),
    style='scientific'
)
```

### Boundary Layer Plots

The `boundary_layer_plots.py` module provides functions for plotting boundary layer profiles:

- `plot_boundary_layer_profile`: Plot boundary layer profile (u+ vs y+)
- `plot_y_plus_distribution`: Plot y+ distribution along the aerofoil surface
- `plot_velocity_profile`: Plot velocity profile at a specific location

Example usage:

```python
from src.visualization.boundary_layer_plots import plot_boundary_layer_profile

# Plot boundary layer profile
fig, ax = plot_boundary_layer_profile(
    y_plus_values=[y_plus1, y_plus2, y_plus3],
    u_plus_values=[u_plus1, u_plus2, u_plus3],
    labels=['Model 1', 'Model 2', 'Model 3'],
    title="Boundary Layer Profile",
    style='scientific',
    show_analytical=True
)
```

### Convergence Plots

The `convergence_plots.py` module provides functions for plotting convergence history:

- `plot_convergence_history`: Plot convergence history of residuals
- `plot_convergence_comparison`: Plot convergence history comparison for multiple turbulence models
- `plot_iteration_time`: Plot iteration time for one or more simulations

Example usage:

```python
from src.visualization.convergence_plots import plot_convergence_history

# Plot convergence history
fig, ax = plot_convergence_history(
    iterations=iterations,
    residuals=residuals,
    parameter_names=['Continuity', 'X-Velocity', 'Y-Velocity', 'k', 'epsilon'],
    title="Convergence History",
    style='scientific',
    convergence_threshold=1e-4
)
```

### Contour Plots

The `contour_plots.py` module provides functions for creating contour and vector field plots:

- `plot_contour`: Create a contour plot of a scalar field
- `plot_vector_field`: Create a vector field plot
- `plot_streamlines`: Create a streamline plot
- `plot_multi_contour`: Create multiple contour plots in a single figure

Example usage:

```python
from src.visualization.contour_plots import plot_contour, plot_vector_field

# Plot contour
fig, ax = plot_contour(
    x=x_grid,
    y=y_grid,
    z=pressure,
    title="Pressure Contour",
    colorbar_label="Pressure (Pa)",
    style='scientific'
)

# Plot vector field
fig, ax = plot_vector_field(
    x=x_grid,
    y=y_grid,
    u=u_velocity,
    v=v_velocity,
    title="Velocity Field",
    add_magnitude=True
)
```

## ANSYS Fluent Integration

The tool is specifically designed to work with ANSYS Fluent simulations. I've implemented several features to make this integration seamless:

### Exporting Data from Fluent

1. In Fluent, go to File > Export > Solution Data
2. Select CSV format and choose the parameters you need:
   - Pressure (static, total, dynamic)
   - Velocity components and magnitude
   - Turbulence quantities (k, epsilon, omega, Reynolds stresses)
   - Wall data (shear stress, y+)
   - Density and viscosity

3. For wall-bounded flows, make sure to include:
   - Wall shear stress
   - Near-wall cell distance
   - Velocity parallel to the wall

### Reading Fluent Data

The `FLUENTFileReader` class automatically handles Fluent's naming conventions and data structure:

```python
from src.data_extraction.file_readers import FLUENTFileReader

# Read data from a Fluent export
reader = FLUENTFileReader('fluent_export.csv')
data = reader.read()

# Access specific data
pressure = data['pressure']
velocity = data['velocity']['magnitude']
wall_shear = data['wall']['shear_stress']
```

### Working with Fluent Mesh Data

For more complex geometries, you might need to export the mesh data separately:

1. In Fluent, use File > Export > Mesh to export the mesh data
2. Use the `extract_surface_data()` method to work with specific boundaries

## F1 Aerodynamics Applications

This tool is particularly useful for F1 aerodynamics simulations. Here are some specific applications:

### Analyzing Downforce and Drag

```python
from src.data_processing.pressure_calcs import calculate_pressure_forces, calculate_lift_drag_coefficients

# Calculate forces on a wing surface
Fx, Fy = calculate_pressure_forces(
    x=wing_x_coords,
    y=wing_y_coords,
    pressure=wing_pressure,
    reference_pressure=freestream_pressure
)

# Calculate lift and drag coefficients
Cl, Cd = calculate_lift_drag_coefficients(
    Fx=Fx,
    Fy=Fy,
    dynamic_pressure=dynamic_pressure,
    reference_area=wing_area
)
```

### Comparing Turbulence Models for F1 Applications

Different turbulence models can significantly affect the prediction of flow separation and reattachment, which are critical for F1 aerodynamics:

```python
from src.data_processing.comparison_utils import compare_turbulence_models

# Compare results from different turbulence models
comparison = compare_turbulence_models(
    model_results={
        'Standard k-epsilon': ke_data,
        'Realizable k-epsilon': rke_data,
        'Reynolds Stress Model': rsm_data
    },
    parameter='pressure_coefficient',
    reference_model='Reynolds Stress Model'
)
```

### Visualizing Flow Structures

For analyzing complex flow structures around F1 cars:

```python
from src.visualization.contour_plots import plot_streamlines

# Plot streamlines around a car component
fig, ax = plot_streamlines(
    x=grid_x,
    y=grid_y,
    u=velocity_x,
    v=velocity_y,
    title="Flow Structure Around Front Wing",
    add_magnitude=True,
    density=2
)
```

## Examples

### Example 1: Analyzing Pressure Coefficient Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from src.data_extraction.file_readers import FLUENTFileReader, ExperimentalDataReader
from src.data_processing.pressure_calcs import calculate_pressure_coefficient
from src.visualization.pressure_plots import plot_pressure_coefficient

# Load simulation data
sim_reader = FLUENTFileReader('path/to/fluent_output.csv')
sim_data = sim_reader.read()

# Load experimental data
exp_reader = ExperimentalDataReader('path/to/experimental_data.csv')
exp_data = exp_reader.read()

# Extract coordinates and pressure values
x = sim_data['coordinates']['x']
pressure = sim_data['pressure']
reference_pressure = 101325.0  # Pa
dynamic_pressure = 0.5 * sim_data['density'] * sim_data['velocity']['magnitude']**2

# Calculate pressure coefficient
cp = calculate_pressure_coefficient(pressure, reference_pressure, dynamic_pressure)

# Extract experimental data
x_exp = exp_data['pressure_coefficient']['x/c']
cp_exp = exp_data['pressure_coefficient']['Cp']

# Plot pressure coefficient distribution
fig, ax = plot_pressure_coefficient(
    x_coords=[x],
    cp_values=[cp],
    labels=['CFD Simulation'],
    title="Pressure Coefficient Distribution",
    experimental_data=(x_exp, cp_exp),
    style='scientific'
)

# Save the figure
plt.savefig('pressure_coefficient.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 2: Comparing Boundary Layer Profiles from Different Turbulence Models

```python
import numpy as np
import matplotlib.pyplot as plt
from src.data_extraction.file_readers import FLUENTFileReader
from src.data_processing.turbulence_calcs import calculate_u_tau, calculate_y_plus, calculate_u_plus
from src.visualization.boundary_layer_plots import plot_boundary_layer_profile

# Load simulation data for different turbulence models
reader1 = FLUENTFileReader('path/to/standard_wall_function.csv')
data1 = reader1.read()

reader2 = FLUENTFileReader('path/to/enhanced_wall_function.csv')
data2 = reader2.read()

reader3 = FLUENTFileReader('path/to/reynolds_stress_model.csv')
data3 = reader3.read()

# Extract data for each model
models = [data1, data2, data3]
model_names = ['Standard Wall Function', 'Enhanced Wall Function', 'Reynolds Stress Model']
y_plus_values = []
u_plus_values = []

for data in models:
    # Extract wall data
    wall_shear_stress = data['wall']['shear_stress']
    density = data['density']
    viscosity = 1.8e-5  # kg/(m*s)
    distance = data['coordinates']['y']  # Distance from wall
    velocity = data['velocity']['magnitude']
    
    # Calculate u_tau
    u_tau = calculate_u_tau(density, wall_shear_stress)
    
    # Calculate y+
    y_plus = calculate_y_plus(distance, u_tau, density, viscosity)
    
    # Calculate u+
    u_plus = calculate_u_plus(velocity, u_tau)
    
    y_plus_values.append(y_plus)
    u_plus_values.append(u_plus)

# Plot boundary layer profiles
fig, ax = plot_boundary_layer_profile(
    y_plus_values=y_plus_values,
    u_plus_values=u_plus_values,
    labels=model_names,
    title="Boundary Layer Profile Comparison",
    style='scientific',
    show_analytical=True
)

# Save the figure
plt.savefig('boundary_layer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## API Reference

For detailed API reference, see the docstrings in each module file.
