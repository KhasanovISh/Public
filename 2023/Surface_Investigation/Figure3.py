# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:00:03 2023

@author: Leon
Source code for drawing figure 3 in the journal article
https://
DOI
"""

from plotting import *
import numpy as np


def plot_3():
    
    WAVELENGTH_UM = 125
    angles = np.linspace(14,26,100)
    d_range = np.linspace(1,100,100)
    plot_Agrand(angles, d_range, WAVELENGTH_UM*um,100, (25,40))
    angles = np.linspace(14,35,100)
    plot_SPR(angles, WAVELENGTH_UM*um, 3 , [33, 33, 32, 33, 32, 33])