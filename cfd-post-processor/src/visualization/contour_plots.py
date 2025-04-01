"""
Functions for making plots of flow fields in CFD.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from ..visualization.plot_utils import set_plot_style, add_annotations, add_legend, add_colorbar


def plot_contour(x, y, z, title="Contour Plot", x_label="x", y_label="y", 
                colorbar_label="Value", style='scientific', fig_size=(10, 6), 
                save_path=None, levels=20, cmap='jet', show_grid=True, 
                alpha=1.0, add_lines=True, line_color='k', line_width=0.5, 
                line_alpha=0.5, add_labels=False, label_fmt='%.2f', 
                label_fontsize=8):
    """Makes a plot showing how a value changes across a 2D area."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y
    
    contourf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, alpha=alpha)
    
    if add_lines:
        contour_lines = ax.contour(X, Y, z, levels=levels, colors=line_color, 
                                 linewidths=line_width, alpha=line_alpha)
        if add_labels:
            ax.clabel(contour_lines, inline=True, fontsize=label_fontsize, 
                     fmt=label_fmt)
    
    add_colorbar(fig, ax, contourf, label=colorbar_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(show_grid)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_vector_field(x, y, u, v, title="Vector Field", x_label="x", y_label="y",
                     style='scientific', fig_size=(10, 6), save_path=None,
                     show_grid=True, color='k', scale=1.0, width=0.005,
                     density=1, add_magnitude=True, cmap='jet'):
    """Makes a plot showing how velocity changes across a 2D area."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y
    
    if add_magnitude:
        magnitude = np.sqrt(u**2 + v**2)
        quiver = ax.quiver(X[::density, ::density], Y[::density, ::density],
                          u[::density, ::density], v[::density, ::density],
                          magnitude[::density, ::density],
                          cmap=cmap, scale=scale, width=width)
        add_colorbar(fig, ax, quiver, label="Velocity Magnitude")
    else:
        ax.quiver(X[::density, ::density], Y[::density, ::density],
                 u[::density, ::density], v[::density, ::density],
                 color=color, scale=scale, width=width)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(show_grid)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_streamlines(x, y, u, v, title="Streamlines", x_label="x", y_label="y",
                    style='scientific', fig_size=(10, 6), save_path=None,
                    show_grid=True, color='k', density=1, linewidth=1,
                    add_magnitude=True, cmap='jet', arrowsize=1):
    """Makes a plot showing the flow paths."""
    set_plot_style(style=style, figure_size=fig_size)
    fig, ax = plt.subplots()
    
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y
    
    if add_magnitude:
        magnitude = np.sqrt(u**2 + v**2)
        streamplot = ax.streamplot(X, Y, u, v, density=density, color=magnitude,
                                 cmap=cmap, linewidth=linewidth, arrowsize=arrowsize)
        add_colorbar(fig, ax, streamplot.lines, label="Velocity Magnitude")
    else:
        ax.streamplot(X, Y, u, v, density=density, color=color,
                    linewidth=linewidth, arrowsize=arrowsize)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(show_grid)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_multi_contour(x, y, z_list, titles, main_title="Multi-Contour Plot", 
                      x_label="x", y_label="y", colorbar_label="Value",
                      style='scientific', fig_size=(15, 10), save_path=None,
                      levels=20, cmap='jet', show_grid=True, alpha=1.0,
                      subplot_layout=None, share_colorbar=True):
    """Makes a plot showing multiple values across a 2D area."""
    set_plot_style(style=style, figure_size=fig_size)
    
    n_plots = len(z_list)
    if subplot_layout is None:
        if n_plots <= 3:
            subplot_layout = (1, n_plots)
        else:
            subplot_layout = (2, (n_plots + 1) // 2)
    
    rows, cols = subplot_layout
    fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y
    
    if share_colorbar:
        vmin = min(np.min(z) for z in z_list)
        vmax = max(np.max(z) for z in z_list)
    
    contour_plots = []
    for i, (z, title) in enumerate(zip(z_list, titles)):
        if i < len(axes.flat):
            ax = axes.flat[i]
            
            if share_colorbar:
                contourf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            else:
                contourf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, alpha=alpha)
            
            contour_plots.append(contourf)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(show_grid)
            ax.set_aspect('equal')
    
    for i in range(n_plots, len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    if share_colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(contour_plots[0], cax=cbar_ax)
        cbar.set_label(colorbar_label)
    else:
        for i, contourf in enumerate(contour_plots):
            ax = axes.flat[i]
            cbar = fig.colorbar(contourf, ax=ax)
            if i % cols == cols - 1:
                cbar.set_label(colorbar_label)
    
    fig.suptitle(main_title, fontsize=16)
    
    if share_colorbar:
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
