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
from helper import *
from settings import *

# general imports
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
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
        image = np.load(path) #[:,:,0]
        # np.save(path, image)
        start = 200
        curve = image[700][start:1100]
        positions.append(np.mean(np.where(curve == np.min(curve))[0]+start))
        # plt.plot(np.arange(len(curve)),curve)
    # plt.show()
    plt.plot(wavelengths, positions)
    return numbers, wavelengths, positions

def experimental_gradient_SPR():
    asbolute_path = os.path.join(os.path.dirname(__file__),"data")
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

def minimize_unit(SPR_unit, upper_limit):
    bnds = (SPR_unit.TIR(), upper_limit)
    get_R = lambda x: SPR_unit.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

def minimize_unit_unbounded(SPR_unit):
    bnds = (0, 90)
    get_R = lambda x: SPR_unit.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

def setup_SPR_unit(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    AuSPR.layers[2] = Layer(lambda x: gradient_func(x), d_nm*nm)
    return AuSPR

def setup_SPR_unit_wet(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_SPR_unit(wavelength_nm, d_nm, gradient_func)
    AuSPR.layers[3] = Layer(get_RI_by_name(wavelength_nm, "Ethanol"), 100)
    return AuSPR

def get_SPR_minima_gradient_dry(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_SPR_unit(wavelength_nm, d_nm, gradient_func)
    return minimize_unit(AuSPR, 60)

def get_SPR_minima_gradient_dry_all(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_SPR_unit(wavelength_nm, d_nm, gradient_func)
    return minimize_unit_unbounded(AuSPR)

def get_SPR_minima_gradient_wet_all(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_SPR_unit_wet(wavelength_nm, d_nm, gradient_func)
    return minimize_unit_unbounded(AuSPR)

def get_SPR_minima_gradient_wet(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_SPR_unit_wet(wavelength_nm, d_nm, gradient_func)    
    return minimize_unit(AuSPR, 90)



def get_TIR():
    AuSPR = setup_AuSPR_theoretical(600, d_nm= 0, n= 1)
    return AuSPR.TIR()

@memoize_to_file()
def calculate_data():
    return experimental_gradient_SPR()



# @memoize_to_file()
def calculate_optmization_data(wls, ths, func):
    TIR = get_TIR()

    thetas_func = []
    for i, N in enumerate(range(4)):
        thetas = []
        for wl in wls:
            thetas.append(func(wl, 220, FUNCTIONS[N]))
        plt.show()
        thetas = np.array(thetas)
        # thetas[thetas <= TIR+0.1] = None
        thetas_func.append(thetas)
    return thetas_func

# def calculate_optmization_data_dry():