# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 04:08:34 2023

@author: Leon

Source codes for drawing figure in the article for the conference WECONF-2023
[WECONF - Международная научная конференция](https://weconf-guap.ru/)

"""

from settings import *
from model import *
from helper import *

import matplotlib.pyplot as plt

def plot_and_save_figure_wet_measurement():
    wavelengths, angles = calculate_data() 
    wavelengths = np.linspace(590,600,5)
    thetas_func = calculate_optmization_data(wavelengths, angles, 
                                             get_SPR_minima_gradient_wet)
    
    for i, thetas in enumerate(thetas_func):
        plt.plot(wavelengths, thetas, 
                 FUNCTION_STYLES[i], 
                 label = FUNCTION_LABELS[i])

    plt.xlabel(WAVELENGTH_LABEL)
    plt.ylabel(RESONANCE_ANGLE_LABEL)
    plt.legend()
    
    savefig("wet_measurement")
    
    plt.show()