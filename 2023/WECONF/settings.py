# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:02:24 2023

@author: Leon
"""



IMAGE_DPI = 400

FUNCTION_STYLES = ["r--","g--","b--","k--"]
FUNCTION_LABELS = ["1","2","3","4"]

FUNCTIONS = [lambda x: x*0+1.65476709,
              lambda x: -0.49352205* x + 1.89009383,
              lambda x: -0.16717102 *x*x  +1.70482002,
              lambda x: -0.0722967 *x*x*x + 1.67035575
              ]

WAVELENGTH_LABEL = "Длина волны λ, нм"
RESONANCE_ANGLE_LABEL = "Резонансный угол θ, $^{o}$"
ANGLE_LABEL = "углы θ, $^{o}$"
REFLECTIVITY_LABEL = "Коэф-т отражения $R_p$"

EXPERIMENTAL_POINTS_LABEL = "0"
EXPERIMENTAL_POINTS_STYLE = "ko"
EXPERIMENTAL_ERRORBAR_COLOR = "k"

