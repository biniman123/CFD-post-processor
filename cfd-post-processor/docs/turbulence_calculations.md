# Turbulence Calculations

This document provides detailed information about the turbulence calculations implemented in the CFD post-processor.

## Overview

The turbulence calculations module provides functions for computing key parameters used in CFD analysis, particularly for wall-bounded flows and boundary layer analysis.

## Functions

### `calculate_u_tau`

Calculates the friction velocity (u_tau) from density and wall shear stress:

```
u_tau = sqrt(tau_w / rho)
```

where:
- tau_w is the wall shear stress
- rho is the fluid density

### `calculate_y_plus`

Computes the dimensionless wall distance (y+):

```
y+ = (y * u_tau * rho) / mu
```

This parameter is crucial for:
- Assessing mesh quality near walls
- Determining appropriate wall treatment in turbulence models
- Validating boundary layer resolution

### `calculate_law_of_wall`

Implements the analytical law of the wall, which consists of three regions:

1. Viscous sublayer (y+ < 5):
   ```
   u+ = y+
   ```

2. Buffer layer (5 ≤ y+ ≤ 30):
   - Uses a blending function for smooth transition between viscous and log-law regions

3. Log-law region (y+ > 30):
   ```
   u+ = (1/kappa) * ln(y+) + B
   ```
   where:
   - kappa = 0.41 (von Karman constant)
   - B = 5.5 (log-law constant)

### `calculate_spalding_law`

Implements Spalding's law of the wall, which provides a single formula valid for all regions of the boundary layer:

```
y+ = u+ + exp(-kappa*B) * (exp(kappa*u+) - 1 - kappa*u+ - (kappa*u+)^2/2 - (kappa*u+)^3/6)
```

This formulation is often more accurate than the piecewise law of the wall in the buffer region.

## Usage Examples

```python
from src.data_processing.turbulence_calcs import calculate_y_plus, calculate_u_plus

# Calculate y+ for boundary layer analysis
y_plus = calculate_y_plus(
    distance=0.0001,  # m
    u_tau=1.2,        # m/s
    density=1.225,    # kg/m^3
    viscosity=1.8e-5  # kg/(m*s)
)

# Compare with law of the wall
u_plus = calculate_u_plus(velocity=20.0, u_tau=1.2)
u_plus_theory = calculate_law_of_wall(y_plus)
```

## Notes

- All functions handle both scalar and array inputs
- NaN and inf values are replaced with zeros to ensure numerical stability
- The implementation uses numpy for efficient array operations 