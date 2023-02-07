# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:05:55 2022

@author: Leon
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import imageio

from tqdm import tqdm

import dataclasses

import traceback

from scipy.signal import argrelextrema
from PIL import Image
from scipy import interpolate
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import dual_annealing
from SPPPy import ExperimentSPR, Layer, MaterialDispersion
import seaborn as sns

nm = 1e-9
um = 1e-6

def get_glass_RI(wavelength_nm, filename):
    glass_dispersion = np.genfromtxt(filename)
    glass_wl, glass_n = list(zip(*glass_dispersion))
    return np.interp(wavelength_nm/1000, glass_wl, glass_n)

def setup_AgSPR_theoretical(wavelength_nm, d_nm):
    glass_refractive_index = get_glass_RI(wavelength_nm, 
                                          'Glass LK-7 dispersion.dat')
    # n_Ag = 0.15 + 4.819j
    d_Ag = 55
    AgSPR = ExperimentSPR()
    AgSPR.wavelength = wavelength_nm * nm
    AgSPR.add(Layer(glass_refractive_index, 1))
    # AgSPR.add(Layer(n_Ag, d_Ag*1e-9))
    AgSPR.add(Layer(MaterialDispersion("Ag"), d_Ag * nm))
    AgSPR.add(Layer(1.45, d_nm * nm))
    AgSPR.add(Layer(1, 1))
    return AgSPR

# def find_SPR_minima(R)
#     res = minimize_scalar(minimizeSPR_AgSPR, method='bounded', bounds=bnds)

def load_experimental_SPR(i):
    folders = ['12 prism glass 273mm p-pol',
    '11 prism 0nm 273mm p-pol',
    '16 prism 5nm 273mm p-pol',
    '13 prism 13nm 273mm p-pol',
    '15 prism 17nm 273mm p-pol',
    '14 prism 30nm 273mm p-pol']
    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 480
    
    angles = np.linspace(43.1,47.1,IMAGE_WIDTH)
    # angles = np.flip(angles)
    
    wls = np.load("results\\"+"wavelengths " + folders[i] + ".npy")
    wls = np.flip(wls)
    img = Image.open("results\\"+"reflection " + folders[i] + " 2.bmp")
    r = np.array(img)/255
    # plt.imshow(r)
    # plt.show()
    
    return wls, angles, r



if __name__ == '__main__':
    
    
    def get_SPR_minima(wavelength_nm, d_nm):
        AgSPR = setup_AgSPR_theoretical(wavelength_nm, d_nm)
        angles = np.linspace(43,47,1000)
        bnds = (AgSPR.TIR(),55)
        get_R = lambda x: AgSPR.R(angles=[x])[0]
        res = minimize_scalar(get_R,
                        method='bounded', bounds=bnds)
        return res.x
    
    def get_SPR_minima_r(wavelength_nm, d_nm):
        AgSPR = setup_AgSPR_theoretical(wavelength_nm, d_nm)
        angles = np.linspace(43,47,1000)
        bnds = (AgSPR.TIR(),55)
        # plt.title(f"λ = {wavelength_nm} d = {d_nm}")
        # plt.plot(AgSPR.R(angles= angles))
        # plt.show()
        
        get_R = lambda x: AgSPR.R(angles=[x])[0]
        res = minimize_scalar(get_R,
                        method='bounded', bounds=bnds)
        return get_R(res.x)
    
    def plot_SPR_minima(i, d_nm, text_label = "", dy = 0, low_wl = 500, color = 'k'):
        wavelengths, angles, r = load_experimental_SPR(i)
        
        low_index = 0
        up_index = len(wavelengths)
        for idx, wl in enumerate(wavelengths):
            if wl < low_wl:
                up_index = idx
                break
        
        theta_min_exp = np.array([angles[np.where(r[i] == np.min(r[i]))[0]][0]
                          for i in range(len(wavelengths))])
        
        dyfit = np.ones(len(wavelengths[low_index:up_index]))*0.17
        
        plt.fill_between(wavelengths[low_index:up_index],
                         theta_min_exp[low_index:up_index] - dyfit, 
                         theta_min_exp[low_index:up_index] + dyfit,
                         alpha=0.2, color = color)
    
        plt.plot(wavelengths[low_index:up_index], theta_min_exp[low_index:up_index],
                  linestyle='-', label=f"{d_nm} нм", color = color)
        plt.text(752, theta_min_exp[0] + dy, text_label)

    def plot_SPR_minima_r(i, d_nm, text_label = "", dy = 0, low_wl = 500, color = 'k'):
        wavelengths, angles, r = load_experimental_SPR(i)
        
        low_index = 0
        up_index = len(wavelengths)
        for idx, wl in enumerate(wavelengths):
            if wl < low_wl:
                up_index = idx
                break
        r_min_exp  = []
        for i in range(len(r)):
            r_min_exp.append(np.min(r[i]))
        r_min_exp = np.array(r_min_exp)
        dyfit = np.ones(len(wavelengths[low_index:up_index]))*0.05
        
        plt.fill_between(wavelengths[low_index:up_index],
                         r_min_exp[low_index:up_index] - dyfit, 
                         r_min_exp[low_index:up_index] + dyfit,
                         alpha=0.2, color = color)
    
        plt.plot(wavelengths[low_index:up_index], r_min_exp[low_index:up_index],
                  linestyle='-', label=f"{d_nm} нм", color= color)
        plt.text(752, r_min_exp[0]+dy, text_label)
        
    def plot_SPR_minima_theory(i, d_nm, text_label = "", color = 'k'):
        wavelengths, angles, r = load_experimental_SPR(i)
        theta_min_theor = [get_SPR_minima(wl, d_nm) for wl in wavelengths]
        plt.plot(wavelengths, theta_min_theor, 
                  linestyle='--', label=f"{d_nm} нм", color = color)
    
    def plot_SPR_minima_r_theory(i, d_nm, text_label = "", color = 'k'):
        wavelengths, angles, r = load_experimental_SPR(i)
        theta_min_theor = [get_SPR_minima_r(wl, d_nm) for wl in wavelengths]
        plt.plot(wavelengths, theta_min_theor, 
                  linestyle='--', label=f"{d_nm} нм", color= color)
        
    # -------------------------------------------
    
    sns.set_theme()
    sns.set_context("paper")
    sns.axes_style("ticks")
    
    plt.rcParams.update({'font.size': 14})
    # plot_SPR_minima(1,0,1, -0.1, color = 'r')
    # plot_SPR_minima(3,8,2, -0.1, color = 'g')
    # plot_SPR_minima(4,12,3, 0.1, color = 'b')
    plot_SPR_minima(5,25,4,0, 630, color = 'm')
    
    # plot_SPR_minima_theory(1,0,1, color = 'r')
    #plot_SPR_minima_theory(3,10.2,2, color = 'g')
    # plot_SPR_minima_theory(4,12,3, color = 'b')
    plot_SPR_minima_theory(5,27.4,4, color = 'm')
    

    
    
    plt.ylim(43,47)
    plt.xlim(515,765)
    plt.ylabel(r"Resonance angle, $^o$")
    plt.xlabel("Wavelength λ, nm")
    sns.set_style("ticks")
    sns.despine()
    # plt.legend()
    # plt.savefig("Minima Angle SPR.jpg", 
    #             dpi=400, 
    #             bbox_inches='tight')
    
    plt.show()  
    
    # -------------------------------------------
    
    # plt.xlim(515,765)
    # plot_SPR_minima_r_theory(1,0,1, color = 'r')
    # plot_SPR_minima_r_theory(3,9,2, color = 'b')
    # plot_SPR_minima_r_theory(4,12,3, color = 'g')
    # plot_SPR_minima_r_theory(5,26,4, color = 'm')
    
    # plot_SPR_minima_r(1,0,1,0.005, color = 'r')
    # plot_SPR_minima_r(3,8,2, color = 'b')
    # plot_SPR_minima_r(4,12,3,-0.01, color = 'g')
    # plot_SPR_minima_r(5,25,4,0, 630, color = 'm')
    # plt.ylabel(r"Resonance minima $R_p$")
    # plt.xlabel("Wavelength λ, nm")
    # # sns.set_context("paper")
    # sns.set_style("ticks")
    # sns.despine()
    # # plt.legend()
    # plt.savefig("Minima R SPR.jpg", 
    #             dpi=400, 
    #             bbox_inches='tight')
    
    # plt.show()    
    
    # -------------------------------------------
    
    # angles = np.linspace(43,47,1000)
    # AgSPR = setup_AgSPR_theoretical(515)
    # R = AgSPR.R(angles = angles)
    # plt.plot(angles, R, label="515")
    # AgSPR = setup_AgSPR_theoretical(750)
    # R = AgSPR.R(angles = angles)
    # plt.plot(angles, R, label="750")
    # plt.legend()
    # plt.show()
    
    
    
    
    
    # print(res.x)
    
    # angles = np.sort(np.append(angles, res.x))
    # R = np.insert(R, np.where(angles == res.x)[0], get_R(res.x))
    
    # plt.plot(angles, R)
    # plt.show()