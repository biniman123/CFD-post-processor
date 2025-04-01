"""
Convergence Plots Module

This module contains functions for plotting convergence history data from CFD simulations.
"""

import matplotlib.pyplot as plt
import numpy as np
from ..visualization.plot_utils import set_plot_style, add_annotations, add_legend


def plot_convergence_history(iterations, residuals, parameter_names=None,
                            title="Convergence History", 
                            x_label="Iterations", y_label="Residuals",
                            style='scientific', fig_size=(12, 6), save_path=None,
                            show_grid=True, log_scale_y=True, colors=None, 
                            line_styles=None, convergence_threshold=None):
    """Plot convergence history of residuals.
    
    Args:
        iterations (array): Array of iteration numbers
        residuals (list): List of residual arrays for each parameter
        parameter_names (list, optional): List of parameter names
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        log_scale_y (bool, optional): Whether to use logarithmic scale for y-axis
        colors (list, optional): List of colors for each parameter
        line_styles (list, optional): List of line styles for each parameter
        convergence_threshold (float, optional): Convergence threshold to highlight
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Default colors and line styles if not provided
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # Plot each residual
    for i, residual in enumerate(residuals):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        label = parameter_names[i] if parameter_names is not None and i < len(parameter_names) else f"Parameter {i+1}"
        
        ax.plot(iterations, residual, label=label, color=color, linestyle=line_style, linewidth=1.5)
    
    # Add convergence threshold if provided
    if convergence_threshold is not None:
        ax.axhline(y=convergence_threshold, color='k', linestyle='--', 
                  linewidth=1.5, label=f'Convergence Threshold: {convergence_threshold:.0e}')
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Set y-axis to log scale if requested
    if log_scale_y:
        ax.set_yscale('log')
    
    # Show grid
    ax.grid(show_grid)
    
    # Add legend
    add_legend(ax, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_convergence_comparison(iterations_list, residuals_list, model_names, parameter_name,
                               title="Convergence Comparison", 
                               x_label="Iterations", y_label="Residuals",
                               style='scientific', fig_size=(12, 6), save_path=None,
                               show_grid=True, log_scale_y=True, colors=None, 
                               line_styles=None, convergence_threshold=None):
    """Plot convergence history comparison for multiple turbulence models.
    
    Args:
        iterations_list (list): List of iteration arrays for each model
        residuals_list (list): List of residual arrays for each model
        model_names (list): List of model names
        parameter_name (str): Name of the parameter to compare
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        log_scale_y (bool, optional): Whether to use logarithmic scale for y-axis
        colors (list, optional): List of colors for each model
        line_styles (list, optional): List of line styles for each model
        convergence_threshold (float, optional): Convergence threshold to highlight
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Default colors and line styles if not provided
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    # Plot each model's convergence history
    for i, (iterations, residuals, model_name) in enumerate(zip(iterations_list, residuals_list, model_names)):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        ax.plot(iterations, residuals, label=model_name, color=color, linestyle=line_style, linewidth=1.5)
    
    # Add convergence threshold if provided
    if convergence_threshold is not None:
        ax.axhline(y=convergence_threshold, color='k', linestyle='--', 
                  linewidth=1.5, label=f'Convergence Threshold: {convergence_threshold:.0e}')
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} - {parameter_name}")
    
    # Set y-axis to log scale if requested
    if log_scale_y:
        ax.set_yscale('log')
    
    # Show grid
    ax.grid(show_grid)
    
    # Add legend
    add_legend(ax, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_iteration_time(iterations, time_values, model_names=None,
                       title="Iteration Time", 
                       x_label="Iterations", y_label="Cumulative Time (s)",
                       style='scientific', fig_size=(10, 6), save_path=None,
                       show_grid=True, colors=None, line_styles=None):
    """Plot iteration time for one or more simulations.
    
    Args:
        iterations (list): List of iteration arrays for each model
        time_values (list): List of time arrays for each model
        model_names (list, optional): List of model names
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        colors (list, optional): List of colors for each model
        line_styles (list, optional): List of line styles for each model
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Default colors and line styles if not provided
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    # Ensure iterations and time_values are lists of arrays
    if not isinstance(iterations[0], (list, np.ndarray)):
        iterations = [iterations]
        time_values = [time_values]
    
    # Plot each model's iteration time
    for i, (iters, times) in enumerate(zip(iterations, time_values)):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        label = model_names[i] if model_names is not None and i < len(model_names) else f"Model {i+1}"
        
        ax.plot(iters, times, label=label, color=color, linestyle=line_style, linewidth=1.5)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Show grid
    ax.grid(show_grid)
    
    # Add legend
    add_legend(ax, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
