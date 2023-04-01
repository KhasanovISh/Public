# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:23:44 2023

@author: Leon
"""
import os
import numpy as np

def get_glass_RI(wavelength_nm, filename):
    data_path = os.path.join(os.getcwd(), 'optics', filename)
    glass_dispersion = np.genfromtxt(data_path)
    glass_wl, glass_n = list(zip(*glass_dispersion))
    return np.interp(wavelength_nm/1000, glass_wl, glass_n)

def get_LK7_RI(wavelength_nm):
    n = get_glass_RI(wavelength_nm, 'Glass LK-7 dispersion.dat')
    return n

def get_BK7_RI(wavelength_nm):
    x = wavelength_nm/1000
    n=(1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**.5
    return n
    
def get_air_RI(wavelength_nm):
    return 1



# def get_linear(x):
#     return 1.5 - 0.5*x

# def get_glass_RI(wavelength_nm, filename):
#     glass_dispersion = np.genfromtxt(filename)
#     glass_wl, glass_n = list(zip(*glass_dispersion))
#     return np.interp(wavelength_nm/1000, glass_wl, glass_n)

# def get_prism_RI(wavelength_nm):
#     x = wavelength_nm/1000
#     # n=(1+3.00+1.90/(1-0.113/x**2))**.5
#     # n = 1
#     n=(1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**.5
#     return n

# def get_medium_RI(wavelength_nm):
#     x = wavelength_nm/1000
#     # n= 1.386820+17856.021E-6*x**-2
#     # n = 1.9
#     n = 1
#     return n