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

@memoize_to_file()
def calculate_data():
    return experimental_gradient_SPR()

@memoize_to_file()
def calculate_optmization_data(wls, ths):
    TIR = get_TIR()
    
    thetas_func = []
    for i, N in enumerate(range(4)):
        thetas = []
        for wl in wls:
            thetas.append(get_SPR_minima_gradient2(wl, 220, FUNCTIONS[N]))
        thetas = np.array(thetas)
        thetas[thetas <= TIR+0.1] = None
        thetas_func.append(thetas)
    
    return thetas_func

def plot_and_save_figure_dry_measurement():
    wavelengths, angles = calculate_data()
    plt.plot(wavelengths, angles, 
              EXPERIMENTAL_POINTS_STYLE, 
              label = EXPERIMENTAL_POINTS_LABEL)
    # plt.show()
    
    thetas_func = calculate_optmization_data(wavelengths, angles)
    
    for i, thetas in enumerate(thetas_func):
        plt.plot(wavelengths, thetas, 
                 FUNCTION_STYLES[i], 
                 label = FUNCTION_LABELS[i])

    plt.errorbar(wavelengths, angles, 
                 yerr=0.15, capsize=2,
                 color=EXPEIMENTAL_ERRORBAR_COLOR, fmt = "o")
    plt.plot(wavelengths, angles, EXPERIMENTAL_POINTS_STYLE)
    plt.xlabel(WAVELENGTH_LABEL)
    plt.ylabel(RESONANCE_ANGLE_LABEL)
    plt.legend()
    
    savefig("dry_measurement")
    
    plt.show()