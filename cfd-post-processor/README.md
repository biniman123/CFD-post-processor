# CFD Post-Processing Tool

A comprehensive Python tool for post-processing Computational Fluid Dynamics (CFD) simulation results, with a primary focus on airfoil analysis while maintaining versatility for other aerodynamic applications.

## Features

- **Data Extraction**: Read and process data from various CFD simulation output formats, including ANSYS Fluent
- **Turbulence Analysis**: Calculate and analyze key parameters like Y+, U+, and pressure coefficients
- **Model Comparison**: Compare results from different turbulence models (standard wall function, enhanced wall function, Reynolds Stress Model)
- **Visualization**: Generate publication-quality plots for:
  - Pressure coefficient distributions
  - Boundary layer profiles
  - Convergence history
  - Contour and vector field plots

## Use Cases

This tool was designed with versatility in mind, supporting a range of CFD post-processing needs:

### Primary Use Case: Airfoil Analysis
- Pressure coefficient distribution along airfoil surfaces
- Boundary layer development and separation analysis
- Comparison of different turbulence models for airfoil simulations
- Lift and drag coefficient calculations

### Additional Applications:
- **F1 & Automotive Aerodynamics**: Wing elements, diffusers, and full car analysis
- **External Aerodynamics**: Building aerodynamics, wind engineering
- **Internal Flows**: Pipe flows, heat exchangers, and HVAC systems
- **Academic Research**: Validation of simulation results against experimental data
- **Educational Use**: Visualization of fundamental fluid dynamics concepts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cfd-post-processor.git
cd cfd-post-processor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import numpy as np
import matplotlib.pyplot as plt
from src.data_extraction.file_readers import FLUENTFileReader
from src.data_processing.turbulence_calcs import calculate_y_plus, calculate_u_plus
from src.visualization.pressure_plots import plot_pressure_coefficient

# Load simulation data
reader = FLUENTFileReader('path/to/fluent_output.csv')
data = reader.read()

# Process data
# ... (data processing code)

# Create visualization
fig, ax = plot_pressure_coefficient(
    x_coords, 
    cp_values,
    labels=['Standard Wall Function'],
    title="Pressure Coefficient Distribution"
)
plt.show()
```

### Advanced Usage

See the [documentation](docs/README.md) for detailed usage examples and API reference.

## ANSYS Fluent Compatibility

This post-processor is specifically designed to work with ANSYS Fluent simulations. To use with Fluent:

1. **Exporting Data from Fluent**:
   - In Fluent, go to File > Export > Solution Data
   - Select CSV format and choose the parameters you need (pressure, velocity, turbulence quantities, etc.)
   - Ensure wall data is included if you plan to analyze boundary layers

2. **Using with Airfoil Simulations**:
   - Export data along the airfoil surface for pressure coefficient analysis
   - Create line/rake data perpendicular to the airfoil surface for boundary layer analysis
   - Use the `extract_airfoil_data()` function to automatically process airfoil surface data

3. **Performance Tips**:
   - For complex geometries, consider exporting data for specific sections separately
   - Use the contour plots to identify key flow structures around the airfoil
   - Compare different turbulence models to ensure accurate prediction of separation points

## Project Structure

```
cfd-post-processor/
├── src/
│   ├── data_extraction/       # Data extraction modules
│   ├── data_processing/       # Data processing modules
│   └── visualization/         # Visualization modules
├── tests/                     # Test scripts and sample data
├── docs/                      # Documentation
├── examples/                  # Example scripts
└── requirements.txt           # Dependencies
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
