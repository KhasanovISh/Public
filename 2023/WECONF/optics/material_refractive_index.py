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

def get_ethanol_RI(wavelength_nm):
    x = wavelength_nm/1000
    # n = get_LK7_RI(wavelength_nm)
    # n=1.35265+0.00306*x**-2+0.00002*x**-4
    # n=1.294611+12706.403E-6*x**-2
    # n=1.384715+4960.733E-6*x**-2 # Butanol
    n= 1.47797+0.00598*x**-2-0.00036*x**-4
    # n = 1.4298
    # n = 1.5
    # n = n=(1+0.75831/(1-0.01007/x**2)+0.08495/(1-8.91377/x**2))**.5
    return n

def get_RI_by_name(wavelength_nm, name):
    materials = {
        "Glycerol": lambda x: 1.45797+0.00598*x**-2-0.00036*x**-4,
        "Ethanol": lambda x: 1.35265+0.00306*x**-2+0.00002*x**-4,
        "Benzyl": lambda x: 0*x + 1.54049
        }
    if name in materials:
        x = wavelength_nm/1000
        n = materials[name](x)
        return n

def get_ZrO2_RI(wavelength_nm):
    x = wavelength_nm/1000
    return (1+1.347091/(1-(0.062543/x)**2)+2.117788/(1-(0.166739/x)**2)+9.452943/(1-(24.320570/x)**2))**.5

def get_TiO2_RI(wavelength_nm):
    x = wavelength_nm/1000
    return (5.913+0.2441/(x**2-0.0803))**.5

def get_SiO2_RI(wavelength_nm):
    x = wavelength_nm/1000
    return (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5