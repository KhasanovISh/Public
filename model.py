# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:08:50 2023

@author: Leon
"""

# local imports
import sys
sys.path.append('E:\Python\SPPPy (1)')
from SPPPy import ExperimentSPR, Layer, MaterialDispersion
from optics import *

# general imports
from pathlib import Path

import numpy as np
# from PIL import Image
# from Helper2 import horizontal_slice_image, running_mean, hdr2

# imports packages for optimization problem
from scipy.optimize import minimize_scalar
# from functools import partial
# import seaborn as sns
# import pybobyqa

def setup_AuSPR_theoretical(wavelength_nm, d_nm = 0, n = 1):
    glass_refractive_index = get_LK7_RI(wavelength_nm)
    # n_Ag = 0.15 + 4.819j
    d_Au = 50
    AgSPR = ExperimentSPR(polarisation='p')
    AgSPR.wavelength = wavelength_nm * nm
    AgSPR.add(Layer(glass_refractive_index, 1))
    # AgSPR.add(Layer(n_Ag, d_Ag*1e-9))
    AgSPR.add(Layer(MaterialDispersion("Au"), d_Au * nm))
    AgSPR.add(Layer(n, d_nm * nm))
    AgSPR.add(Layer(1, 1))
    return AgSPR

def get_SPR_minima(wavelength_nm, d_nm = 0, refractive_index = 1):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= refractive_index)
    # angles = np.linspace(AuSPR.TIR(),60,1000)
    bnds = (AuSPR.TIR(),60)
    get_R = lambda x: AuSPR.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

def get_linear_scale_params(x,y):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def get_wavelength(folder, number):
    spectrum = np.genfromtxt(folder.joinpath(f"{number}.DAT"), skip_header = 7)
    return spectrum[np.where(spectrum[:,1] == np.max(spectrum[:,1]))[0][0]][0]

def get_spectral_angle_curve(folder):
    folder = Path(folder)
    numbers = np.arange(68,82) #93
    exposures = np.array([1000, 3205, 6024, 10000])
    exposure = exposures[3]
    wavelengths=[]
    positions = []
    for number in numbers:
        wavelengths.append(get_wavelength(folder, number))
        path = folder.joinpath(f"{number}_experiment_exposure_{exposure}.npy")
        image = np.load(path)[:,:,0]
        start = 200
        curve = image[700][start:1100]
        positions.append(np.mean(np.where(curve == np.min(curve))[0]+start))
    return numbers, wavelengths, positions

def experimental_gradient_SPR():
    asbolute_path = r"E:\Experiments\2022-12-17"
    folders = ['Au 2 0nm', 
               'Au 2 grad']
    
    
    folder = Path(asbolute_path).joinpath(folders[0])
    numbers, wavelengths, positions = get_spectral_angle_curve(folder)
    
    
    mes_grad_numbers = np.array([77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88])
    mes_grad_positions = np.array([444, 459, 476, 507, 530, 566, 599, 631, 671, 747, 784, 810])    
    
    thetas2 = []
    for wl in wavelengths:
        thetas2.append(get_SPR_minima(wl))
   
    m,c = get_linear_scale_params(positions, thetas2)
    # plt.plot(positions, thetas, 'o', label='Original data', markersize=10)
    # plt.plot(positions, m*np.array(positions) + c, 'r', label='Fitted line')
    # plt.legend()
    
    mes_grad_wavelengths = []
    for num in mes_grad_numbers:
        wl = get_wavelength(folder, num)
        mes_grad_wavelengths.append(wl)
    
    # plt.plot(wavelengths, thetas2,  label = "1: Au")
    # plt.plot(mes_grad_wavelengths, m*mes_grad_positions + c, "b.", label = "2: Au+град.")
    # plt.xlabel("Длина волны λ, нм")
    # plt.ylabel("Резонансный угол θ, $^{o}$") 
    # plt.legend()
    # # savefig("Измерения")
    # plt.show()
    print("Измерения")
    return mes_grad_wavelengths, m*mes_grad_positions + c