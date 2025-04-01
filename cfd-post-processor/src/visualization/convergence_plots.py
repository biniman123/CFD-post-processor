"""
Functions for making plots of how CFD runs converge.
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
    """Makes a plot of how the residuals change over time."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    for i, residual in enumerate(residuals):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        label = parameter_names[i] if parameter_names is not None and i < len(parameter_names) else f"Parameter {i+1}"
        
        ax.plot(iterations, residual, label=label, color=color, linestyle=line_style, linewidth=1.5)
    
    if convergence_threshold is not None:
        ax.axhline(y=convergence_threshold, color='k', linestyle='--', 
                  linewidth=1.5, label=f'Convergence Threshold: {convergence_threshold:.0e}')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if log_scale_y:
        ax.set_yscale('log')
    
    ax.grid(show_grid)
    add_legend(ax, loc='best')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_convergence_comparison(iterations_list, residuals_list, model_names, parameter_name,
                               title="Convergence Comparison", 
                               x_label="Iterations", y_label="Residuals",
                               style='scientific', fig_size=(12, 6), save_path=None,
                               show_grid=True, log_scale_y=True, colors=None, 
                               line_styles=None, convergence_threshold=None):
    """Makes a plot comparing how different turbulence models converge."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    for i, (iterations, residuals, model_name) in enumerate(zip(iterations_list, residuals_list, model_names)):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        ax.plot(iterations, residuals, label=model_name, color=color, linestyle=line_style, linewidth=1.5)
    
    if convergence_threshold is not None:
        ax.axhline(y=convergence_threshold, color='k', linestyle='--', 
                  linewidth=1.5, label=f'Convergence Threshold: {convergence_threshold:.0e}')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} - {parameter_name}")
    
    if log_scale_y:
        ax.set_yscale('log')
    
    ax.grid(show_grid)
    add_legend(ax, loc='best')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_iteration_time(iterations, time_values, model_names=None,
                       title="Iteration Time", 
                       x_label="Iterations", y_label="Cumulative Time (s)",
                       style='scientific', fig_size=(10, 6), save_path=None,
                       show_grid=True, colors=None, line_styles=None):
    """Makes a plot of how long each iteration takes."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    if not isinstance(iterations[0], (list, np.ndarray)):
        iterations = [iterations]
        time_values = [time_values]
    
    for i, (iters, times) in enumerate(zip(iterations, time_values)):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        label = model_names[i] if model_names is not None and i < len(model_names) else f"Model {i+1}"
        
        ax.plot(iters, times, label=label, color=color, linestyle=line_style, linewidth=1.5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.grid(show_grid)
    add_legend(ax, loc='best')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
