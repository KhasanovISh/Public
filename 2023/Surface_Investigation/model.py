# -*- coding: utf-8 -*-
"""
Created on Sat Apr 1

Physical model calculated here

@author: Leon
"""

# local imports
import sys
sys.path.append('E:\Python\Common\SPPPy')
from SPPPy import ExperimentSPR, Layer, MaterialDispersion
# from optics import *
from helper import *
from settings import *

# general imports
from pathlib import Path
import os

import numpy as np
from numpy.lib import scimath as SM
import matplotlib.pyplot as plt
# from PIL import Image
# from Helper2 import horizontal_slice_image, running_mean, hdr2

# imports packages for optimization problem
from scipy.optimize import minimize_scalar, root_scalar
# from functools import partial
# import seaborn as sns
import pybobyqa

def inverse_cm_from_m(wavelength: float) -> float:
    """
    Convert a wavelength in meters to inverse centimeters.
    
    Args:
        wavelength (float): The wavelength in meters.
        
    Returns:
        float: The frequency in inverse centimeters of the given wavelength.
    """
    return 0.01 / wavelength

def m_from_inverse_cm(frequency):
    """
    Convert a frequency in inverse centimeters to meters.
    
    Args:
        frequency (float): The frequency in inverse centimeters.
        
    Returns:
        float: The wavelength in meters of the given frequency.
    """
    return 0.01 / frequency

def eps2nk(eps):
    """
    Converts complex dielectric constant `eps` to complex refractive index `n + 1j*k`.
    
    Parameters:
    -----------
    eps: complex
        Complex dielectric constant.
        
    Returns:
    --------
    n + 1j*k: complex
        Complex refractive index where `n` is the real part and `k` is the imaginary part.
    """
    n = np.sqrt((np.abs(eps)+np.real(eps))/2)
    k = np.sqrt((np.abs(eps)-np.real(eps))/2)
    return n + 1j * k
    # return SM.sqrt(eps)

def nk2eps(nk):
    n = np.real(nk)
    k = np.imag(nk)
    epsr = n*n - k*k
    epsi = 2 * n * k
    return epsr + 1j * epsi

def eps_drude(omega, w_p, gamma_p, eps_inf):
    return eps_inf -  w_p*w_p / (omega *(omega + 1.0j * gamma_p))

def w(eps, eps1, theta, wl):
    ns = np.sin(np.pi*theta/180)
    return 2 * np.pi / wl * SM.sqrt(eps - eps1*ns*ns)

def r_ij(eps_i, eps_i1, eps1, theta, wl):
    w_i = w(eps_i,eps1,theta, wl)
    w_i1 = w(eps_i1,eps1,theta, wl)
    return (eps_i1 * w_i - eps_i * w_i1)/(eps_i1 * w_i + eps_i * w_i1)

def optimal_d(eps1,eps2,eps3,theta, wl):
    w1 = w(eps1,eps1,theta, wl)
    w2 = w(eps2,eps1,theta, wl)
    w3 = w(eps3,eps1,theta, wl)
    # r12 = r_ij(eps1, eps2, eps1, theta, wl)
    # r23 = r_ij(eps2, eps3, eps1, theta, wl)
    
    numerator = (eps2 * w1 - eps1 * w2) * (eps3 * w2 + eps2 * w3)
    denominator = (eps3 * w2 - eps2 * w3) * (eps2 * w1 + eps1 * w2)
    return -1j / (2 * w2) * SM.log(-numerator /denominator)

def setup_SPR_structure(wavelength, d=0):
    SPR_structure = ExperimentSPR()
    SPR_structure.wavelength = wavelength
    SPR_structure.add(Layer(3.41, 1))
    SPR_structure.add(Layer(1, d*um))
    eps3 = eps_drude(inverse_cm_from_m(SPR_structure.wavelength), 
                     302.4, 10.3, 15.68)
    SPR_structure.add(Layer(eps2nk(eps3), 1))
    return SPR_structure

def SPR_to_eps(SPR_structure):
    # l = SPR_structure.layers
    # [print(layer) for layer in SPR_structure.layers]
    eps = [nk2eps(layer.n) for layer in SPR_structure.layers.values()]
    return eps

def optimal_gap(SPR_structure, theta):
    eps1, eps2, eps3 = SPR_to_eps(SPR_structure)
    wl = SPR_structure.wavelength
    return optimal_d(eps1,eps2,eps3,theta, wl)


def get_minimal_theta(SPR_structure):
    bnds = (0,90) #SPR_structure.TIR()
    get_R = lambda x: SPR_structure.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    # if res.flag == res.EXIT_SUCCESS:
    # print(res)
    return res.x

def get_minimal_gap(SPR_structure, theta):
    bnds = (0,2*SPR_structure.wavelength/um) #SPR_structure.TIR()
    def get_R(x):
        SPR_structure.layers[1].thickness = x * um
        return SPR_structure.R(angles=[theta])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    # if res.flag == res.EXIT_SUCCESS:
    # print(res)
    return res.x

# def find_minimal_d(SPR_structure, th):
#     # eps = SPR_to_eps(SPR_structure)
#     # wl = SPR_structure.wavelength
#     def reflectivity(d):
#         SPR_structure.layers[1].thickness = d * um
#         return SPR_structure.R(angles=[th])[0] #R(*eps,deg2rad(th), d*um, wl)
#     # tir = np.arcsin(1/np.sqrt(eps[0]))
#     tir = SPR_structure.TIR()
#     d_opt = optimal_gap(SPR_structure, tir) # np.real(optimal_d(*eps, tir, wl)/dim)
#     # result = minimize_scalar(reflectivity, bounds=(tir-2, tir+2), method='bounded')
#     # return result.x
#     lower = 0
#     upper  = 1000
#     # theta_range = np.linspace(lower,upper,1000)
#     # R_range = [R(*eps,deg2rad(th),d_opt*dim, wl) for th in theta_range]
#     # th = theta_range[np.where(R_range == np.min(R_range))]
#     soln = pybobyqa.solve(reflectivity, [d_opt], 
#                           bounds=([lower],[upper]),
#                           maxfun=100, 
#                           seek_global_minimum=True)
#     return soln.x

# def get_optimal_theta(SPR_structure):
#     eps = SPR_to_eps(SPR_structure)
#     wl = SPR_structure.wavelength
#     def imaginary_d(theta):
#         return np.imag(optimal_d(*eps, theta, wl))
#     th_opt = SPR_structure.TIR() #get_tir(eps)
#     if np.sign(imaginary_d(10)) != np.sign(imaginary_d(40)):
#         # print("ищем пересечние Im с 0")
#         sol = root_scalar(imaginary_d, bracket=[10, 40], method='brentq')
#         th_opt = sol.root
#         if not sol.converged:
#             raise 'Решение для резонансного угла не найдено!'
#     return th_opt