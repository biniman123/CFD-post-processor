# Visualization Functions

This document provides detailed information about the visualization functions in the CFD post-processor.

## Contour Plots

The `plot_contour` function creates publication-quality contour plots of scalar fields from CFD simulations.

### Parameters

- `x, y`: Coordinates (1D arrays or 2D grids)
- `z`: Scalar field values (2D grid)
- `title`: Plot title
- `x_label, y_label`: Axis labels
- `colorbar_label`: Label for the colorbar
- `style`: Plot style ('default', 'scientific', 'presentation')
- `fig_size`: Figure size in inches
- `save_path`: Path to save the figure
- `levels`: Number of contour levels or array of level values
- `cmap`: Colormap name
- `show_grid`: Whether to show grid lines
- `alpha`: Alpha transparency for filled contours
- `add_lines`: Whether to add contour lines
- `line_color`: Color for contour lines
- `line_width`: Width for contour lines
- `line_alpha`: Alpha transparency for contour lines
- `add_labels`: Whether to add contour labels
- `label_fmt`: Format string for contour labels
- `label_fontsize`: Font size for contour labels

### Usage Example

```python
from src.visualization.contour_plots import plot_contour

# Create a contour plot of pressure field
fig, ax = plot_contour(
    x=grid_x,
    y=grid_y,
    z=pressure,
    title="Pressure Distribution",
    x_label="x (m)",
    y_label="y (m)",
    colorbar_label="Pressure (Pa)",
    style='scientific',
    add_lines=True,
    add_labels=True
)

# Save the plot
plt.savefig('pressure_distribution.png')
```

## Notes

- The function automatically handles both 1D and 2D coordinate inputs
- Contour lines and labels can be toggled for different visualization needs
- The scientific style is optimized for publication-quality plots
- Grid lines can be shown or hidden based on the visualization context 