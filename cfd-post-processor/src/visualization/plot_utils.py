"""
Helper functions for making CFD plots look nice.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def set_plot_style(style='default', font_size=12, figure_size=(10, 6)):
    """Sets up how the plots should look."""
    style_settings = {}
    
    plt.rcParams['figure.figsize'] = figure_size
    style_settings['figure.figsize'] = figure_size
    
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
    """Saves a figure to files in different formats."""
    if formats is None:
        formats = ['png']
    
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    saved_files = []
    for fmt in formats:
        save_path = f"{filename}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=transparent)
        saved_files.append(save_path)
    
    return saved_files


def add_annotations(ax, annotations, fontsize=10):
    """Adds text labels with arrows to a plot."""
    annotation_objects = []
    
    for anno in annotations:
        text = anno['text']
        xy = anno.get('xy', (0, 0))
        xytext = anno.get('xytext', None)
        
        if xytext is not None:
            annotation = ax.annotate(
                text, xy=xy, xytext=xytext,
                fontsize=fontsize,
                arrowprops=dict(arrowstyle='->', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
            )
        else:
            annotation = ax.annotate(
                text, xy=xy,
                fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
            )
        
        annotation_objects.append(annotation)
    
    return annotation_objects


def create_colormap(cmap_name='jet', min_val=0, max_val=1, num_colors=256):
    """Makes a color map for plotting."""
    cmap = plt.get_cmap(cmap_name, num_colors)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    return cmap, norm


def add_colorbar(fig, ax, mappable, label=None, orientation='vertical', pad=0.05):
    """Adds a color bar to a plot."""
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, pad=pad)
    if label is not None:
        cbar.set_label(label)
    return cbar


def add_legend(ax, loc='best', title=None, frameon=True, framealpha=0.8):
    """Adds a legend to a plot."""
    legend = ax.legend(loc=loc, title=title, frameon=frameon, framealpha=framealpha)
    return legend
