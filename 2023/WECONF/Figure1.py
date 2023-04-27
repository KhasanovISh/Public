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
    
    # #b---
    # #analysis of what resonance minima look like at a particular wavelength
    # #b---
    # for i, thetas in enumerate(thetas_func):
    #     plt.xlabel(WAVELENGTH_LABEL)
    #     plt.ylabel(REFLECTIVITY_LABEL)
    #     min_r = []
    #     for j, theta in enumerate(thetas):
    #         gradSPR = setup_SPR_unit(wavelengths2[j], 220, FUNCTIONS[i])
    #         min_r.append(gradSPR.R(angles = [theta])[0])
    #     plt.plot(wavelengths2, min_r,
    #              FUNCTION_STYLES[i],
    #              label = FUNCTION_LABELS[i])
    # plt.legend()
    # plt.show()
    # #b---
    
    # #a---
    # #analysis of what resonance curves look like at a particular wavelength
    # #a---
    # angle_range = np.linspace(0,90,300)
    # for wavelength in [633]:
    #     plt.xlabel(ANGLE_LABEL)
    #     plt.ylabel(REFLECTIVITY_LABEL)
    #     plt.title(wavelength)
    #     for i, func in enumerate(FUNCTIONS):
    #         gradSPR = setup_SPR_unit(wavelength, 220, func)
    #         plt.plot(angle_range, gradSPR.R(angles = angle_range), FUNCTION_STYLES[i], label = FUNCTION_LABELS[i])
    #     plt.legend()
    #     plt.show()
    # #a---