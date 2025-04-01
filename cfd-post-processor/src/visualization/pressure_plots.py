"""
Functions for making plots of pressure stuff in CFD.
"""

import matplotlib.pyplot as plt
import numpy as np
from ..visualization.plot_utils import set_plot_style, add_annotations, add_legend


def plot_pressure_coefficient(x_coords, cp_values, labels=None, title="Pressure Coefficient Distribution",
                             experimental_data=None, x_label="x/c", y_label="Pressure Coefficient (Cp)",
                             style='scientific', fig_size=(10, 6), save_path=None, show_grid=True,
                             invert_yaxis=True, colors=None, markers=None, line_styles=None):
    """Makes a plot of how pressure coefficient changes along the surface."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if markers is None:
        markers = ['o', 's', '^', 'v', 'D', '*', 'x']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    if not isinstance(x_coords[0], (list, np.ndarray)):
        x_coords = [x_coords]
        cp_values = [cp_values]
    
    for i, (x, cp) in enumerate(zip(x_coords, cp_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        ax.plot(x, cp, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(x)//20), markersize=6)
    
    if experimental_data is not None:
        x_exp, cp_exp = experimental_data
        ax.scatter(x_exp, cp_exp, color='k', marker='o', s=30, label='Experimental Data')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(show_grid)
    
    if invert_yaxis:
        ax.invert_yaxis()
    
    add_legend(ax, loc='best')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_pressure_contour(x, y, pressure, title="Pressure Contour", 
                         x_label="x", y_label="y", colorbar_label="Pressure",
                         style='scientific', fig_size=(10, 6), save_path=None,
                         levels=20, cmap='jet', show_grid=True):
    """Makes a plot showing pressure changes across a 2D area."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    contour = ax.contourf(x, y, pressure, levels=levels, cmap=cmap)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label)
    
    contour_lines = ax.contour(x, y, pressure, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(show_grid)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_pressure_comparison(x_coords, cp_values, model_names, 
                            title="Pressure Coefficient Comparison",
                            experimental_data=None, x_label="x/c", y_label="Pressure Coefficient (Cp)",
                            style='scientific', fig_size=(12, 8), save_path=None,
                            show_grid=True, invert_yaxis=True, subplot_layout=None):
    """Makes a plot comparing pressure coefficients from different turbulence models."""
    set_plot_style(style=style, figure_size=fig_size)
    
    n_models = len(model_names)
    if subplot_layout is None:
        if n_models <= 3:
            subplot_layout = (1, n_models)
        else:
            subplot_layout = (2, (n_models + 1) // 2)
    
    rows, cols = subplot_layout
    fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    for i, (x, cp, model_name) in enumerate(zip(x_coords, cp_values, model_names)):
        if i < len(axes):
            ax = axes[i]
            
            ax.plot(x, cp, 'b-', label=model_name)
            
            if experimental_data is not None:
                x_exp, cp_exp = experimental_data
                ax.scatter(x_exp, cp_exp, color='r', marker='o', s=20, label='Experimental')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{model_name}")
            ax.grid(show_grid)
            
            if invert_yaxis:
                ax.invert_yaxis()
            
            add_legend(ax, loc='best')
    
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
