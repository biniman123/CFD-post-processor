"""
Functions for making plots of boundary layer stuff in CFD.
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
    """Makes a plot of the boundary layer profile (u+ vs y+)."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if markers is None:
        markers = ['o', 's', '^', 'v', 'D', '*', 'x']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    if not isinstance(y_plus_values[0], (list, np.ndarray)):
        y_plus_values = [y_plus_values]
        u_plus_values = [u_plus_values]
    
    for i, (y_plus, u_plus) in enumerate(zip(y_plus_values, u_plus_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        ax.plot(y_plus, u_plus, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(y_plus)//10), markersize=6)
    
    if show_analytical:
        y_plus_range = np.logspace(0, 3, 100)
        
        ax.plot(y_plus_range[y_plus_range < 5], y_plus_range[y_plus_range < 5], 
               'k--', linewidth=1.5, label='Viscous Sublayer: u+ = y+')
        
        kappa = 0.41
        B = 5.5
        log_law = (1/kappa) * np.log(y_plus_range[y_plus_range > 30]) + B
        ax.plot(y_plus_range[y_plus_range > 30], log_law, 
               'k-.', linewidth=1.5, label='Log Law: u+ = (1/κ)ln(y+) + B')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.grid(show_grid)
    add_legend(ax, loc='best')
    plt.tight_layout()
    
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
    """Makes a plot of how y+ changes along the surface."""
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
        y_plus_values = [y_plus_values]
    
    for i, (x, y_plus) in enumerate(zip(x_coords, y_plus_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        ax.plot(x, y_plus, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(x)//20), markersize=6)
    
    if y_plus_target is not None:
        y_min, y_max = y_plus_target
        ax.axhspan(y_min, y_max, alpha=0.2, color='green', label=f'Target y+ range: {y_min}-{y_max}')
    
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


def plot_velocity_profile(y_coords, velocity_values, labels=None, 
                         title="Velocity Profile", 
                         x_label="Velocity", y_label="y",
                         style='scientific', fig_size=(8, 10), save_path=None,
                         show_grid=True, colors=None, 
                         markers=None, line_styles=None,
                         normalize=False):
    """Makes a plot of how velocity changes with height at a spot."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if markers is None:
        markers = ['o', 's', '^', 'v', 'D', '*', 'x']
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    if not isinstance(y_coords[0], (list, np.ndarray)):
        y_coords = [y_coords]
        velocity_values = [velocity_values]
    
    for i, (y, velocity) in enumerate(zip(y_coords, velocity_values)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        label = labels[i] if labels is not None and i < len(labels) else f"Dataset {i+1}"
        
        if normalize:
            velocity = velocity / np.max(np.abs(velocity))
        
        ax.plot(velocity, y, label=label, color=color, marker=marker, linestyle=line_style, 
               markevery=max(1, len(y)//10), markersize=6)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.grid(show_grid)
    add_legend(ax, loc='best')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
