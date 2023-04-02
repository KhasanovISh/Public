# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:00:03 2023

@author: Leon
Source codes for drawing figure in the article for the conference WECONF-2023
[WECONF - Международная научная конференция](https://weconf-guap.ru/)

"""

from settings import *
from model import *
from helper import *

import matplotlib.pyplot as plt


def plot_and_save_figure_dry_measurement():
    wavelengths, angles = calculate_data()
    plt.plot(wavelengths, angles, 
              EXPERIMENTAL_POINTS_STYLE, 
              label = EXPERIMENTAL_POINTS_LABEL)
    # plt.show()
    wavelengths2 = np.linspace(590,670,10)
    thetas_func = calculate_optmization_data(wavelengths2, angles, 
                                             get_SPR_minima_gradient_dry)
    
    for i, thetas in enumerate(thetas_func):
        plt.plot(wavelengths2, thetas, 
                 FUNCTION_STYLES[i], 
                 label = FUNCTION_LABELS[i])

    plt.errorbar(wavelengths, angles, 
                 yerr=0.15, capsize=2,
                 color=EXPERIMENTAL_ERRORBAR_COLOR, fmt = "o")
    plt.plot(wavelengths, angles, EXPERIMENTAL_POINTS_STYLE)
    plt.xlabel(WAVELENGTH_LABEL)
    plt.ylabel(RESONANCE_ANGLE_LABEL)
    plt.legend()
    
    savefig("dry_measurement")
    
    plt.show()