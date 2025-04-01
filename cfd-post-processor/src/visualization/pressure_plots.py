"""
Visualization Module for Pressure Plots

This module contains functions for creating pressure-related visualizations from CFD data.
I've focused on making these plots publication-quality and suitable for F1 aerodynamics analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from ..visualization.plot_utils import set_plot_style, add_annotations, add_legend


def plot_pressure_coefficient(x_coords, cp_values, labels=None, title="Pressure Coefficient Distribution",
                             experimental_data=None, x_label="x/c", y_label="Pressure Coefficient (Cp)",
                             style='scientific', fig_size=(10, 6), save_path=None, show_grid=True,
                             invert_yaxis=True, colors=None, markers=None, line_styles=None):
    """Plot pressure coefficient distribution along x-coordinates.
    
    This is one of the most common plots in aerodynamics analysis. For F1 applications,
    it's particularly useful for analyzing wing sections and diffusers.
    
    Args:
        x_coords (list or array): List of x-coordinate arrays for each dataset
        cp_values (list or array): List of pressure coefficient arrays for each dataset
        labels (list, optional): List of labels for each dataset
        title (str, optional): Plot title
        experimental_data (tuple, optional): Tuple of (x_exp, cp_exp) for experimental data
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        invert_yaxis (bool, optional): Whether to invert the y-axis
        colors (list, optional): List of colors for each dataset
        markers (list, optional): List of markers for each dataset
        line_styles (list, optional): List of line styles for each dataset
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Default colors, markers, and line styles if not provided
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if markers is None:
        markers = ['o', 's', '^', 'v', 'D', '*', 'x']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    # Ensure x_coords and cp_values are lists of arrays
    if not isinstance(x_coords[0], (list, np.ndarray)):
        x_coords = [x_coords]
        cp_values = [cp_values]
    
    # Plot each dataset
    for i, (x, cp) in enumerate(zip(x_coords, cp_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        ax.plot(x, cp, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(x)//20), markersize=6)
    
    # Add experimental data if provided
    if experimental_data is not None:
        x_exp, cp_exp = experimental_data
        ax.scatter(x_exp, cp_exp, color='k', marker='o', s=30, label='Experimental Data')
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Show grid
    ax.grid(show_grid)
    
    # Invert y-axis for pressure coefficient (conventional representation)
    if invert_yaxis:
        ax.invert_yaxis()
    
    # Add legend
    add_legend(ax, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_pressure_contour(x, y, pressure, title="Pressure Contour", 
                         x_label="x", y_label="y", colorbar_label="Pressure",
                         style='scientific', fig_size=(10, 6), save_path=None,
                         levels=20, cmap='jet', show_grid=True):
    """Plot pressure contour on a 2D domain.
    
    For F1 applications, this is useful for visualizing pressure distributions
    around complex geometries like front wings and diffusers.
    
    Args:
        x (array): x-coordinates (2D grid)
        y (array): y-coordinates (2D grid)
        pressure (array): Pressure values (2D grid)
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        colorbar_label (str, optional): Label for colorbar
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        levels (int or array, optional): Number of contour levels or array of level values
        cmap (str, optional): Colormap name
        show_grid (bool, optional): Whether to show grid lines
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Create contour plot
    contour = ax.contourf(x, y, pressure, levels=levels, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Add contour lines
    contour_lines = ax.contour(x, y, pressure, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
    
    # Add contour labels (optional, can be commented out if too cluttered)
    # plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Show grid
    ax.grid(show_grid)
    
    # Set aspect ratio to equal for proper visualization
    ax.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_pressure_comparison(x_coords, cp_values, model_names, 
                            title="Pressure Coefficient Comparison",
                            experimental_data=None, x_label="x/c", y_label="Pressure Coefficient (Cp)",
                            style='scientific', fig_size=(12, 8), save_path=None,
                            show_grid=True, invert_yaxis=True, subplot_layout=None):
    """Plot pressure coefficient comparison for multiple turbulence models.
    
    I've found this particularly useful when comparing different turbulence models
    for F1 aerodynamics, where model selection can significantly impact results.
    
    Args:
        x_coords (list): List of x-coordinate arrays for each model
        cp_values (list): List of pressure coefficient arrays for each model
        model_names (list): List of model names
        title (str, optional): Main plot title
        experimental_data (tuple, optional): Tuple of (x_exp, cp_exp) for experimental data
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        invert_yaxis (bool, optional): Whether to invert the y-axis
        subplot_layout (tuple, optional): Layout of subplots (rows, cols)
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Determine subplot layout
    n_models = len(model_names)
    if subplot_layout is None:
        if n_models <= 3:
            subplot_layout = (1, n_models)
        else:
            subplot_layout = (2, (n_models + 1) // 2)
    
    rows, cols = subplot_layout
    
    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=True)
    
    # Flatten axes array for easy iteration
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    # Plot each model in a separate subplot
    for i, (x, cp, model_name) in enumerate(zip(x_coords, cp_values, model_names)):
        if i < len(axes):
            ax = axes[i]
            
            # Plot model data
            ax.plot(x, cp, 'b-', label=model_name)
            
            # Add experimental data if provided
            if experimental_data is not None:
                x_exp, cp_exp = experimental_data
                ax.scatter(x_exp, cp_exp, color='r', marker='o', s=20, label='Experimental')
            
            # Set labels
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{model_name}")
            
            # Show grid
            ax.grid(show_grid)
            
            # Invert y-axis
            if invert_yaxis:
                ax.invert_yaxis()
            
            # Add legend
            add_legend(ax, loc='best')
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    # Add main title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
