"""
Contour Plots Module

This module contains functions for creating contour plots of CFD simulation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from ..visualization.plot_utils import set_plot_style, add_annotations, add_legend, add_colorbar


def plot_contour(x, y, z, title="Contour Plot", 
                x_label="x", y_label="y", colorbar_label="Value",
                style='scientific', fig_size=(10, 6), save_path=None,
                levels=20, cmap='jet', show_grid=True, alpha=1.0,
                add_lines=True, line_color='k', line_width=0.5, line_alpha=0.5,
                add_labels=False, label_fmt='%.2f', label_fontsize=8):
    """Create a contour plot of a scalar field.
    
    Args:
        x (array): x-coordinates (2D grid or 1D array)
        y (array): y-coordinates (2D grid or 1D array)
        z (array): Scalar field values (2D grid)
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
        alpha (float, optional): Alpha transparency for filled contours
        add_lines (bool, optional): Whether to add contour lines
        line_color (str, optional): Color for contour lines
        line_width (float, optional): Width for contour lines
        line_alpha (float, optional): Alpha transparency for contour lines
        add_labels (bool, optional): Whether to add contour labels
        label_fmt (str, optional): Format string for contour labels
        label_fontsize (int, optional): Font size for contour labels
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Check if x and y are 1D or 2D
    if x.ndim == 1 and y.ndim == 1:
        # Create 2D meshgrid from 1D arrays
        X, Y = np.meshgrid(x, y)
    else:
        # Use provided 2D arrays
        X, Y = x, y
    
    # Create filled contour plot
    contourf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, alpha=alpha)
    
    # Add contour lines if requested
    if add_lines:
        contour_lines = ax.contour(X, Y, z, levels=levels, colors=line_color, 
                                  linewidths=line_width, alpha=line_alpha)
        
        # Add contour labels if requested
        if add_labels:
            ax.clabel(contour_lines, inline=True, fontsize=label_fontsize, fmt=label_fmt)
    
    # Add colorbar
    cbar = add_colorbar(fig, ax, contourf, label=colorbar_label)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Show grid
    ax.grid(show_grid)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_vector_field(x, y, u, v, title="Vector Field", 
                     x_label="x", y_label="y",
                     style='scientific', fig_size=(10, 6), save_path=None,
                     show_grid=True, color='k', scale=1.0, width=0.005,
                     density=1, add_magnitude=True, cmap='jet'):
    """Create a vector field plot.
    
    Args:
        x (array): x-coordinates (2D grid or 1D array)
        y (array): y-coordinates (2D grid or 1D array)
        u (array): x-component of vector field (2D grid)
        v (array): y-component of vector field (2D grid)
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        color (str, optional): Color for vectors (ignored if add_magnitude is True)
        scale (float, optional): Scaling factor for vector lengths
        width (float, optional): Width of vectors
        density (int or tuple, optional): Density of vectors (higher means fewer vectors)
        add_magnitude (bool, optional): Whether to color vectors by magnitude
        cmap (str, optional): Colormap name for vector magnitude
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Check if x and y are 1D or 2D
    if x.ndim == 1 and y.ndim == 1:
        # Create 2D meshgrid from 1D arrays
        X, Y = np.meshgrid(x, y)
    else:
        # Use provided 2D arrays
        X, Y = x, y
    
    # Calculate vector magnitude if coloring by magnitude
    if add_magnitude:
        magnitude = np.sqrt(u**2 + v**2)
        
        # Create quiver plot with magnitude coloring
        quiver = ax.quiver(X[::density, ::density], Y[::density, ::density],
                          u[::density, ::density], v[::density, ::density],
                          magnitude[::density, ::density],
                          cmap=cmap, scale=scale, width=width)
        
        # Add colorbar
        cbar = add_colorbar(fig, ax, quiver, label="Velocity Magnitude")
    else:
        # Create quiver plot with uniform color
        ax.quiver(X[::density, ::density], Y[::density, ::density],
                 u[::density, ::density], v[::density, ::density],
                 color=color, scale=scale, width=width)
    
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


def plot_streamlines(x, y, u, v, title="Streamlines", 
                    x_label="x", y_label="y",
                    style='scientific', fig_size=(10, 6), save_path=None,
                    show_grid=True, color='k', density=1, linewidth=1,
                    add_magnitude=True, cmap='jet', arrowsize=1):
    """Create a streamline plot.
    
    Args:
        x (array): x-coordinates (2D grid or 1D array)
        y (array): y-coordinates (2D grid or 1D array)
        u (array): x-component of vector field (2D grid)
        v (array): y-component of vector field (2D grid)
        title (str, optional): Plot title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        show_grid (bool, optional): Whether to show grid lines
        color (str, optional): Color for streamlines (ignored if add_magnitude is True)
        density (float or tuple, optional): Density of streamlines
        linewidth (float, optional): Width of streamlines
        add_magnitude (bool, optional): Whether to color streamlines by magnitude
        cmap (str, optional): Colormap name for streamline magnitude
        arrowsize (float, optional): Size of arrows on streamlines
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Check if x and y are 1D or 2D
    if x.ndim == 1 and y.ndim == 1:
        # Create 2D meshgrid from 1D arrays
        X, Y = np.meshgrid(x, y)
    else:
        # Use provided 2D arrays
        X, Y = x, y
    
    # Calculate vector magnitude if coloring by magnitude
    if add_magnitude:
        magnitude = np.sqrt(u**2 + v**2)
        
        # Create streamplot with magnitude coloring
        streamplot = ax.streamplot(X, Y, u, v, density=density, color=magnitude,
                                 cmap=cmap, linewidth=linewidth, arrowsize=arrowsize)
        
        # Add colorbar
        cbar = add_colorbar(fig, ax, streamplot.lines, label="Velocity Magnitude")
    else:
        # Create streamplot with uniform color
        ax.streamplot(X, Y, u, v, density=density, color=color,
                    linewidth=linewidth, arrowsize=arrowsize)
    
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


def plot_multi_contour(x, y, z_list, titles, 
                      main_title="Multi-Contour Plot", 
                      x_label="x", y_label="y", colorbar_label="Value",
                      style='scientific', fig_size=(15, 10), save_path=None,
                      levels=20, cmap='jet', show_grid=True, alpha=1.0,
                      subplot_layout=None, share_colorbar=True):
    """Create multiple contour plots in a single figure.
    
    Args:
        x (array): x-coordinates (2D grid or 1D array)
        y (array): y-coordinates (2D grid or 1D array)
        z_list (list): List of scalar field arrays (2D grids)
        titles (list): List of subplot titles
        main_title (str, optional): Main figure title
        x_label (str, optional): Label for x-axis
        y_label (str, optional): Label for y-axis
        colorbar_label (str, optional): Label for colorbar
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        fig_size (tuple, optional): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure
        levels (int or array, optional): Number of contour levels or array of level values
        cmap (str, optional): Colormap name
        show_grid (bool, optional): Whether to show grid lines
        alpha (float, optional): Alpha transparency for filled contours
        subplot_layout (tuple, optional): Layout of subplots (rows, cols)
        share_colorbar (bool, optional): Whether to use a single shared colorbar
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    # Set plot style
    set_plot_style(style=style, figure_size=fig_size)
    
    # Determine subplot layout
    n_plots = len(z_list)
    if subplot_layout is None:
        if n_plots <= 3:
            subplot_layout = (1, n_plots)
        else:
            subplot_layout = (2, (n_plots + 1) // 2)
    
    rows, cols = subplot_layout
    
    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=True)
    
    # Flatten axes array for easy iteration
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    # Check if x and y are 1D or 2D
    if x.ndim == 1 and y.ndim == 1:
        # Create 2D meshgrid from 1D arrays
        X, Y = np.meshgrid(x, y)
    else:
        # Use provided 2D arrays
        X, Y = x, y
    
    # Find global min and max for shared colorbar
    if share_colorbar:
        vmin = min(np.min(z) for z in z_list)
        vmax = max(np.max(z) for z in z_list)
    
    # Create contour plots
    contour_plots = []
    for i, (z, title) in enumerate(zip(z_list, titles)):
        if i < len(axes.flat):
            ax = axes.flat[i]
            
            # Set colorbar range if shared
            if share_colorbar:
                contourf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            else:
                contourf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, alpha=alpha)
            
            contour_plots.append(contourf)
            
            # Set labels and title
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            
            # Show grid
            ax.grid(show_grid)
            
            # Set aspect ratio to equal for proper visualization
            ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    # Add colorbar
    if share_colorbar:
        # Add a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(contour_plots[0], cax=cbar_ax)
        cbar.set_label(colorbar_label)
    else:
        # Add individual colorbars for each subplot
        for i, contourf in enumerate(contour_plots):
            ax = axes.flat[i]
            cbar = fig.colorbar(contourf, ax=ax)
            if i % cols == cols - 1:  # Only add label to rightmost colorbars
                cbar.set_label(colorbar_label)
    
    # Add main title
    fig.suptitle(main_title, fontsize=16)
    
    # Adjust layout
    if share_colorbar:
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Leave space for colorbar and title
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
