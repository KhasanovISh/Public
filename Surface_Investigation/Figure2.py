# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:00:03 2023

@author: Leon
Source code for drawing figure 2 in the journal article
https://
DOI
"""

from plotting import *
import numpy as np


def plot_2():
    
    WAVELENGTH_UM = 159
    angles = np.linspace(14,26,100)
    d_range = np.linspace(1,100,100)
    plot_Agrand(angles, d_range, WAVELENGTH_UM*um,160, (25,50))

    plot_SPR(angles, WAVELENGTH_UM*um, 54.4 , [20.8, 19.3, 20, 20, 20, 20])