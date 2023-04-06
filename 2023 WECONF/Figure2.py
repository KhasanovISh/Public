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
    wavelengths = np.linspace(590,670,10)
    thetas_func = calculate_optmization_data(wavelengths, angles, 
                                             get_SPR_minima_gradient_wet_all)
    
    for i, thetas in enumerate(thetas_func):
        plt.plot(wavelengths, thetas, 
                 FUNCTION_STYLES[i], 
                 label = FUNCTION_LABELS[i])

    plt.xlabel(WAVELENGTH_LABEL)
    plt.ylabel(RESONANCE_ANGLE_LABEL)
    plt.legend()
    
    savefig("wet_measurement")
    plt.show()
    
    # #b---
    # #analysis of resonance mininma curves 
    # #b---
    # for i, thetas in enumerate(thetas_func):
    #     plt.xlabel(WAVELENGTH_LABEL)
    #     plt.ylabel(REFLECTIVITY_LABEL)
    #     min_r = []
    #     for j, theta in enumerate(thetas):
    #         gradSPR = setup_SPR_unit(wavelengths[j], 220, FUNCTIONS[i])
    #         min_r.append(gradSPR.R(angles = [theta])[0])
    #     plt.plot(wavelengths, min_r,
    #              FUNCTION_STYLES[i],
    #              label = FUNCTION_LABELS[i])
    # plt.legend()
    # plt.show()
    # #b---
    
    # #b---
    # #analysis of resonance mininma curves 
    # #b---
    # angle_range = np.linspace(0,90,100)
    # for i, func in enumerate(FUNCTIONS):
    #     plt.xlabel(ANGLE_LABEL)
    #     plt.ylabel(REFLECTIVITY_LABEL)
    #     plt.title(FUNCTION_LABELS[i])
    #     for wavelength in wavelengths:
    #         gradSPR = setup_SPR_unit_wet(wavelength, 220, func)
    #         plt.plot(angle_range, gradSPR.R(angles = angle_range))
    #     plt.legend()
    #     plt.show()
    # #b---
    
    # #a---
    # #analysis of what resonance angles look like at a particular wavelength
    # #a---
    # angle_range = np.linspace(50,70,500)
    # for wavelength in wavelengths:
    #     plt.xlabel(ANGLE_LABEL)
    #     plt.ylabel(REFLECTIVITY_LABEL)
    #     plt.title(wavelength)
    #     for i, func in enumerate(FUNCTIONS):
    #         gradSPR = setup_SPR_unit_wet(wavelength, 220, func)
    #         plt.plot(angle_range, gradSPR.R(angles = angle_range), label = FUNCTION_LABELS[i])
    #     plt.legend()
    #     plt.show()
    # #a---