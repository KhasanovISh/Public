# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 03:56:09 2023

@author: Leon
"""

import sys
sys.path.append('E:\Python\SPPPy (1)\SPPPy')

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from SPPPy import ExperimentSPR, Layer, MaterialDispersion
from scipy.interpolate import interp1d, splrep, splev
from scipy.optimize import minimize, minimize_scalar

import pybobyqa

import material_refractive_index as material

nm = 1e-9

# def setup_unit(wavelength_nm):
#     Unit = ExperimentSPR()
#     Unit.wavelength = wavelength_nm*nm
#     Unit.add(Layer(get_prism_RI(wavelength_nm), 1))
#     Unit.add(Layer(MaterialDispersion("Au"), 50*nm))
#     Unit.add(Layer(lambda x: 1.5 - 0.5*x, 220*nm))
#     Unit.add(Layer(get_medium_RI(wavelength_nm), 100))
#     # Unit.add(Layer(1, 100))
#     return Unit

# def setup_unit2(wavelength_nm, n):
#     Unit = ExperimentSPR()
#     Unit.wavelength = wavelength_nm*nm
#     Unit.add(Layer(get_prism_RI(wavelength_nm), 1))
#     Unit.add(Layer(MaterialDispersion("Au"), 50*nm))
#     Unit.add(Layer(n, 200*nm))
#     Unit.add(Layer(1.2, 10*nm))
#     Unit.add(Layer(get_medium_RI(wavelength_nm), 100))
#     # Unit.add(Layer(1, 100))
#     return Unit


def setup_spr_unit(wavelength_nm, prism_func, layer_func, medium_func):
    Unit = ExperimentSPR()
    Unit.wavelength = wavelength_nm*nm
    Unit.add(Layer(prism_func(wavelength_nm), 1))
    Unit.add(Layer(MaterialDispersion("Au"), 50*nm))
    if callable(layer_func):
        Unit.add(Layer(lambda x: layer_func(x), 220*nm))
    else:
        Unit.add(Layer(layer_func, 220*nm))
    Unit.add(Layer(medium_func(wavelength_nm), 100))
    return Unit

def setup_spectroscopy_unit(wavelength_nm, medium_func, layer_func, plate_func):
    Unit = ExperimentSPR()
    Unit.wavelength = wavelength_nm*nm
    Unit.add(Layer(medium_func(wavelength_nm), 1))
    # Unit.add(Layer(MaterialDispersion("Au"), 50*nm))
    Unit.add(Layer(lambda x: layer_func(x), 220*nm))
    Unit.add(Layer(plate_func(wavelength_nm), 100))
    return Unit

def get_SPR_angle(unit):
    bnds = (unit.TIR(),90)    
    get_R = lambda x: unit.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x


def find_params():
    degree_range = np.linspace(60, 70, 100)
    unit_base = setup_spr_unit(700, material.get_BK7_RI, 
                               material.get_linear, 
                               material.get_air)
    # r_base = np.array(calculate_reflection(degree_range, unit_base))
    
    angle_base = get_SPR_angle(unit_base)
    print(f"{angle_base=}")
    
    def minimize_spr(x):
        unit_grad = setup_spr_unit(700, material.get_BK7_RI, 
                                   x[0], 
                                   material.get_air)
        # r = np.array(calculate_reflection(degree_range, unit_grad))
        # dr = r-r_base
        # minim = np.sum(dr*dr)
        angle_grad = get_SPR_angle(unit_grad)
        da = (angle_base - angle_grad)
        minim = da*da
        print(f"{x} : {minim}")
        return minim 
    
    x0 = np.array([1.3])
    lower = np.array([1])
    upper = np.array([3])
    bnds = (lower, upper)
    res = pybobyqa.solve(minimize_spr, x0, bounds=bnds, maxfun=500,
                          scaling_within_bounds=True)
    print(f"{res.x=}")
    return res.x

# def find():
#     degree_range = np.linspace(40, 50, 100)
#     unit = setup_unit(632)
#     r = np.array(unit.R(angles=degree_range))
#     unit2 = setup_unit(632)
#     def get_R(n):
#         unit2.layers[2] = Layer(n, 200*nm)
#         r2 = np.array(unit2.R(angles=degree_range))
#         return np.sum((r2-r)*(r2-r))
#     res = minimize_scalar(get_R,
#                     method='bounded', bounds=(1,3))
#     print(res.x)

# def plot_SPR(degrees, r):
#     plt.plot(degrees, r)

def calculate_reflection(degree_range, unit):
    data = unit.R(angles=degree_range)
    return data

if __name__ == '__main__':
    degree_range = np.linspace(60, 70, 1000)
    # def plot_SPR(wavelength_nm):
    #     unit = setup_unit(wavelength_nm)
    #     r = calculate_reflection(degree_range, unit)
    #     label_format = f"{unit.wavelength/nm:.0f}"
    #     plt.plot(degree_range, r, label=label_format)
    # plot_SPR(600)
    # plot_SPR(650)
    # plot_SPR(700)
    # plt.legend()
    # plt.show()
    # find()
    # unit = setup_unit(700)
    # plt.vlines(unit.TIR(),0,1, linestyle='--')
    # r = calculate_reflection(degree_range, unit)
    # plt.plot(degree_range, r, label="grad")
    
    
    params = find_params()
    
    unit_grad = setup_spr_unit(700, material.get_BK7_RI, 
                                material.get_linear, 
                                material.get_air)
    r = calculate_reflection(degree_range, unit_grad)
    plt.plot(degree_range, r, label="grad")
    
    unit_grad = setup_spr_unit(700, material.get_BK7_RI, 
                                params[0], 
                                material.get_air)
    r = calculate_reflection(degree_range, unit_grad)
    plt.plot(degree_range, r, label=f"{params[0]:.2f}")
    
    # unit2 = setup_unit2(700, 1.343)
    # # unit2 = setup_unit2(700, 1.332)
    # r = calculate_reflection(degree_range, unit2)
    # plt.plot(degree_range, r, label="const")
    plt.legend()
    plt.show()
    
    import scipy.integrate as integrate
    n = integrate.quad(material.get_linear, 0, 1)
    print(f"{n=}")