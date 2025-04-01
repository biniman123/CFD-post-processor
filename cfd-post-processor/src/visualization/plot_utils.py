"""
Plot Utilities Module

This module contains utility functions for creating and customizing plots.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def set_plot_style(style='default', font_size=12, figure_size=(10, 6)):
    """Set the style for matplotlib plots.
    
    Args:
        style (str, optional): Plot style ('default', 'scientific', 'presentation')
        font_size (int, optional): Base font size
        figure_size (tuple, optional): Default figure size (width, height) in inches
    
    Returns:
        dict: Dictionary containing the style settings
    """
    # Create a dictionary to store style settings
    style_settings = {}
    
    # Set figure size
    plt.rcParams['figure.figsize'] = figure_size
    style_settings['figure.figsize'] = figure_size
    
    # Set font sizes
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size - 2
    plt.rcParams['ytick.labelsize'] = font_size - 2
    plt.rcParams['legend.fontsize'] = font_size - 2
    
    style_settings['font.size'] = font_size
    style_settings['axes.titlesize'] = font_size + 2
    style_settings['axes.labelsize'] = font_size
    style_settings['xtick.labelsize'] = font_size - 2
    style_settings['ytick.labelsize'] = font_size - 2
    style_settings['legend.fontsize'] = font_size - 2
    
    # Apply specific style settings
    if style == 'scientific':
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
        
        style_settings['style'] = 'scientific'
        style_settings['axes.grid'] = True
        style_settings['grid.linestyle'] = '--'
        style_settings['grid.alpha'] = 0.7
        style_settings['font.family'] = 'serif'
        style_settings['mathtext.fontset'] = 'cm'
    
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.weight'] = 'bold'
        
        style_settings['style'] = 'presentation'
        style_settings['axes.grid'] = True
        style_settings['grid.linestyle'] = '-'
        style_settings['grid.alpha'] = 0.3
        style_settings['font.family'] = 'sans-serif'
        style_settings['font.weight'] = 'bold'
    
    else:  # default
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['grid.alpha'] = 0.5
        
        style_settings['style'] = 'default'
        style_settings['axes.grid'] = True
        style_settings['grid.linestyle'] = ':'
        style_settings['grid.alpha'] = 0.5
    
    return style_settings


def save_figure(fig, filename, dpi=300, formats=None, transparent=False):
    """Save a figure to file in multiple formats.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Base filename (without extension)
        dpi (int, optional): Resolution in dots per inch
        formats (list, optional): List of formats to save (e.g., ['png', 'pdf'])
        transparent (bool, optional): Whether to save with transparent background
    
    Returns:
        list: List of saved filenames
    """
    # Default formats
    if formats is None:
        formats = ['png']
    
    # Ensure directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save in each format
    saved_files = []
    for fmt in formats:
        save_path = f"{filename}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=transparent)
        saved_files.append(save_path)
    
    return saved_files


def add_annotations(ax, annotations, fontsize=10):
    """Add text annotations to a plot.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to add annotations to
        annotations (list): List of annotation dictionaries
                           [{'text': 'Annotation text', 'xy': (x, y), 'xytext': (x_offset, y_offset)}]
        fontsize (int, optional): Font size for annotations
    
    Returns:
        list: List of annotation objects
    """
    annotation_objects = []
    
    for anno in annotations:
        # Extract annotation parameters
        text = anno['text']
        xy = anno.get('xy', (0, 0))
        xytext = anno.get('xytext', None)
        
        # Create annotation
        if xytext is not None:
            # With arrow
            annotation = ax.annotate(
                text, xy=xy, xytext=xytext,
                fontsize=fontsize,
                arrowprops=dict(arrowstyle='->', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
            )
        else:
            # Without arrow
            annotation = ax.annotate(
                text, xy=xy,
                fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
            )
        
        annotation_objects.append(annotation)
    
    return annotation_objects


def create_colormap(cmap_name='jet', min_val=0, max_val=1, num_colors=256):
    """Create a custom colormap for plotting.
    
    Args:
        cmap_name (str, optional): Base colormap name
        min_val (float, optional): Minimum value for normalization
        max_val (float, optional): Maximum value for normalization
        num_colors (int, optional): Number of colors in the colormap
    
    Returns:
        tuple: (colormap, norm) for use in plotting functions
    """
    # Create colormap
    cmap = plt.get_cmap(cmap_name, num_colors)
    
    # Create normalization
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    
    return cmap, norm


def add_colorbar(fig, ax, mappable, label=None, orientation='vertical', pad=0.05):
    """Add a colorbar to a plot.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to add colorbar to
        ax (matplotlib.axes.Axes): Axes to add colorbar to
        mappable: The mappable object (e.g., from plt.contourf)
        label (str, optional): Label for the colorbar
        orientation (str, optional): Orientation of the colorbar ('vertical' or 'horizontal')
        pad (float, optional): Padding between the plot and colorbar
    
    Returns:
        matplotlib.colorbar.Colorbar: The created colorbar
    """
    # Create colorbar
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, pad=pad)
    
    # Add label if provided
    if label is not None:
        cbar.set_label(label)
    
    return cbar


def add_legend(ax, loc='best', title=None, frameon=True, framealpha=0.8):
    """Add a legend to a plot with customized appearance.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to add legend to
        loc (str or int, optional): Legend location
        title (str, optional): Legend title
        frameon (bool, optional): Whether to draw the legend frame
        framealpha (float, optional): Alpha transparency of the legend frame
    
    Returns:
        matplotlib.legend.Legend: The created legend
    """
    # Create legend
    legend = ax.legend(loc=loc, title=title, frameon=frameon, framealpha=framealpha)
    
    return legend
