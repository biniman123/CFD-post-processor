"""
Boundary Layer Plots Module

This module contains functions for plotting boundary layer profiles and related data
from CFD simulations.
"""

import matplotlib.pyplot as plt
import numpy as np
from ..visualization.plot_utils import set_plot_style, add_annotations, add_legend


def plot_boundary_layer_profile(y_plus_values, u_plus_values, labels=None, 
                               title="Boundary Layer Profile", 
                               x_label="y+", y_label="u+",
                               style='scientific', fig_size=(10, 6), save_path=None,
                               show_grid=True, log_scale=True, colors=None, 
                               markers=None, line_styles=None,
                               show_analytical=True):
    """Plot boundary layer profile (u+ vs y+).
    
    Args:
        y_plus_values (list or array): List of y+ arrays for each dataset
        u_plus_values (list or array): List of u+ arrays for each dataset
        labels (list, optional): List of labels for each dataset
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        log_scale (bool, optional): Whether to use logarithmic scale for x-axis
        colors (list, optional): List of colors for each dataset
        markers (list, optional): List of markers for each dataset
        line_styles (list, optional): List of line styles for each dataset
        show_analytical (bool, optional): Whether to show analytical law of the wall
    
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
    
    # Ensure y_plus_values and u_plus_values are lists of arrays
    if not isinstance(y_plus_values[0], (list, np.ndarray)):
        y_plus_values = [y_plus_values]
        u_plus_values = [u_plus_values]
    
    # Plot each dataset
    for i, (y_plus, u_plus) in enumerate(zip(y_plus_values, u_plus_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        ax.plot(y_plus, u_plus, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(y_plus)//10), markersize=6)
    
    # Add analytical curves if requested
    if show_analytical:
        # Generate y+ values for analytical curves
        y_plus_range = np.logspace(0, 3, 100)
        
        # Viscous sublayer: u+ = y+
        ax.plot(y_plus_range[y_plus_range < 5], y_plus_range[y_plus_range < 5], 
               'k--', linewidth=1.5, label='Viscous Sublayer: u+ = y+')
        
        # Log-law region: u+ = (1/0.41) * ln(y+) + 5.5
        kappa = 0.41  # von Karman constant
        B = 5.5       # Log-law constant
        log_law = (1/kappa) * np.log(y_plus_range[y_plus_range > 30]) + B
        ax.plot(y_plus_range[y_plus_range > 30], log_law, 
               'k-.', linewidth=1.5, label='Log Law: u+ = (1/κ)ln(y+) + B')
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Set x-axis to log scale if requested
    if log_scale:
        ax.set_xscale('log')
    
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


def plot_y_plus_distribution(x_coords, y_plus_values, labels=None, 
                            title="Y+ Distribution", 
                            x_label="x/c", y_label="y+",
                            style='scientific', fig_size=(10, 6), save_path=None,
                            show_grid=True, log_scale_y=True, colors=None, 
                            markers=None, line_styles=None,
                            y_plus_target=None):
    """Plot y+ distribution along the aerofoil surface.
    
    Args:
        x_coords (list or array): List of x-coordinate arrays for each dataset
        y_plus_values (list or array): List of y+ arrays for each dataset
        labels (list, optional): List of labels for each dataset
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        log_scale_y (bool, optional): Whether to use logarithmic scale for y-axis
        colors (list, optional): List of colors for each dataset
        markers (list, optional): List of markers for each dataset
        line_styles (list, optional): List of line styles for each dataset
        y_plus_target (tuple, optional): Tuple of (min, max) target y+ range to highlight
    
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
    
    # Ensure x_coords and y_plus_values are lists of arrays
    if not isinstance(x_coords[0], (list, np.ndarray)):
        x_coords = [x_coords]
        y_plus_values = [y_plus_values]
    
    # Plot each dataset
    for i, (x, y_plus) in enumerate(zip(x_coords, y_plus_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        ax.plot(x, y_plus, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(x)//20), markersize=6)
    
    # Add target y+ range if provided
    if y_plus_target is not None:
        y_min, y_max = y_plus_target
        ax.axhspan(y_min, y_max, alpha=0.2, color='green', label=f'Target y+ range: {y_min}-{y_max}')
    
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


def plot_velocity_profile(y_coords, velocity_values, labels=None, 
                         title="Velocity Profile", 
                         x_label="Velocity", y_label="y",
                         style='scientific', fig_size=(8, 10), save_path=None,
                         show_grid=True, colors=None, 
                         markers=None, line_styles=None,
                         normalize=False):
    """Plot velocity profile at a specific location.
    
    Args:
        y_coords (list or array): List of y-coordinate arrays for each dataset
        velocity_values (list or array): List of velocity arrays for each dataset
        labels (list, optional): List of labels for each dataset
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        colors (list, optional): List of colors for each dataset
        markers (list, optional): List of markers for each dataset
        line_styles (list, optional): List of line styles for each dataset
        normalize (bool, optional): Whether to normalize velocity by maximum value
    
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
    
    # Ensure y_coords and velocity_values are lists of arrays
    if not isinstance(y_coords[0], (list, np.ndarray)):
        y_coords = [y_coords]
        velocity_values = [velocity_values]
    
    # Plot each dataset
    for i, (y, vel) in enumerate(zip(y_coords, velocity_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        # Normalize velocity if requested
        if normalize and np.max(vel) > 0:
            vel = vel / np.max(vel)
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        # Note: For velocity profiles, we typically plot y on the vertical axis
        # and velocity on the horizontal axis
        ax.plot(vel, y, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(y)//10), markersize=6)
    
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
