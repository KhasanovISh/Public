# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:11:58 2023

@author: Leon
"""

from settings import *
from model import *
from helper import *

import matplotlib.pyplot as plt
import pandas as pd
from labellines import labelLine, labelLines


def plot_set_style(ax):
    """
    Apply a specific style to a matplotlib Axes object.

    Parameters:
    -----------
    ax : matplotlib Axes
        The Axes object to apply the style to.

    Returns:
    --------
    None
    """
	# Set font family and font size for labels and ticks
    plt.rcParams.update({
        "font.family": "monospace",
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
    })
	
	# Set tick direction and hide right and top spines
    ax.tick_params(axis="both", direction="in")
    ax.spines[["right", "top"]].set_visible(False)

def plot_gaps_on_angles(ax: plt.Axes, gaps: list, angles: list) -> pd.DataFrame:
    """
    Plots the array `gaps` as a function
    of an array `angles` on the provided axis `ax`. Returns a pandas dataframe
    with columns 'Theta deg' and 'gap um', containing the input arrays.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis where the plot will be created.
    gaps : array_like
        A complex array representing the gaps to be plotted.
    angles : array_like
        An array representing the angles to be plotted.

    Returns:
    --------
    df : pandas.DataFrame
        A pandas dataframe with columns 'Theta deg' and 'gap um', containing the
        input arrays.

    Example:
    --------
    >>> gaps = np.array([1, 2, 3])
    >>> angles = np.array([10, 20, 30])
    >>> fig, ax = plt.subplots()
    >>> df = plot_gaps_on_angles(ax, gaps, angles)
    >>> plt.show()

    """
    ax.plot(angles, gaps)
    
    # set x and y axis labels and title
    ax.set_xlabel(r"$\theta,\rm deg$", fontsize=SMALL_SIZE)
    ax.set_ylabel("$d$, um", fontsize=SMALL_SIZE)
    ax.set_title("(a)")

    # create dataframe from angles and gaps and return it
    # return pd.DataFrame({'Theta deg': angles, 'gap um': gaps})

def save_plot_data(filename: str, dataframes: list, sheet_names: list = None) -> None:
    """
    Save multiple pandas dataframes to an Excel file and a plot to a PNG file.

    Args:
        filename (str): Name of the output files.
        dfs (list): List of pandas dataframes to be saved.
        sheet_names (list, optional): Names of the sheets in the Excel file. If not provided, the sheets will be numbered.
    """
	# If sheet_names not provided, the excel sheets will be numbered:
    if sheet_names is None:
        sheet_names = list(map(str, range(len(dataframes))))
    with pd.ExcelWriter(f"{filename}.xlsx") as writer:
        for name, df in zip(sheet_names, dataframes):
            df.to_excel(writer, sheet_name=name)
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')