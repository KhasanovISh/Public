# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:00:03 2023

@author: Leon
Source code for drawing figure 1 in the journal article 
https://
DOI
"""

from plotting import *
import numpy as np


def plot_1():
    
    WAVELENGTH_UM = 141
    angles = np.linspace(14,26,100)
    d_range = np.linspace(1,100,100)
    plot_Agrand(angles, d_range, WAVELENGTH_UM*um,130, (25,45))
    angles = np.linspace(14,30,100)
    plot_SPR(angles, WAVELENGTH_UM*um, 11, [28, 28, 28, 28, 28, 28])