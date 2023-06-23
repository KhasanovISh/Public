# -*- coding: utf-8 -*-
"""
Created on 2023-05-20 10:29:19
@author: I. Khasanov
"""
# Standard library imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imports from the local package
sys.path.append('E:\Python\Common') #TODO The use of absolute paths: The current code imports a module from an absolute path, which is not recommended as it makes the code less portable. It would be better to use a relative path or add the module to the Python path.
from SPPPy import ExperimentSPR, Layer

from model import setup_insb_otto_configuration

sys.path.append('E:\Python\Common\common')
from helper import load_expeimental_column_data
from optics import calc_permettivity_drude_model, eps2nk, nk2eps

um = 1e-6

def setup_scheme():
    InSb_Otto_setup = ExperimentSPR()
    InSb_Otto_setup.wavelength = 197*um
    InSb_Otto_setup.add(Layer(1.53,1))
    InSb_Otto_setup.add(Layer(1,0*um))
    InSb_Otto_setup.add(Layer(1,0*um))
    InSb_Otto_setup.add(Layer(eps2nk(-18.52+3.03j),1))
    return InSb_Otto_setup

def find_width(setup, level, x0):
    tir = setup.TIR()
    print(f"{tir=}")
    angle = minimize(lambda x: (setup.R(angles = x)[0]-level)**2, x0, method='SLSQP', 
                     bounds=[(30,90)])
    return angle.x

def find_resonance_angle(setup, x0):
    tir = setup.TIR()
    print(f"{tir=}")
    # x0 = 40 # setup.TIR() + 1
    critical_angle = minimize(lambda x: (setup.R(angles = x)[0])**2, x0,method='SLSQP',
                              bounds=[(40,90)])
    print(critical_angle.x)
    return critical_angle.x

from scipy.optimize import minimize

def get_SPR_params(gap = 0, thickness = 0, x0 = 40, plotting = True):
    setup = setup_scheme()
    setup.layers[1] = Layer(1, gap)
    setup.layers[2] = Layer(1.5, thickness)
    critical_angle = find_resonance_angle(setup, x0)[0]
    
    dip = setup.R(angles=[critical_angle])[0]
    ceiling = setup.R(angles=[89])[0]
    height = ceiling - dip
    half_dip = height/2 + dip
    left_angle = find_width(setup, half_dip, critical_angle-1)
    right_angle = find_width(setup, half_dip, critical_angle+1)
    width = right_angle - left_angle
    
    if plotting:
        degrees = np.linspace(0,90,100)
        reflections = setup.R(angles = degrees)
        plt.plot(degrees, reflections)
        plt.plot(left_angle, half_dip, marker="o", color='g')
        plt.plot(right_angle, half_dip, marker="o", color='g')
        plt.plot(critical_angle, dip, marker="o", color='r')
        plt.show()
        
    return critical_angle, height, width

def yeatman(x_range, w0, gap, x_0=0):
    visual_debug = True
    k_0 = 1.51e12*2*np.pi / 3e11
    n = 1.5
    ca, ha, wa =  get_SPR_params((gap+20)*um,0, 40, visual_debug)
    a = n * k_0 * np.cos(np.pi/180*ca)
    ka = k_0 * np.sin(np.pi/180*ca)

    aw = a * wa
    aw2 = aw*aw
    Gia = (0.5 * (aw - np.sqrt(aw2 * (1-ha))))[0]
    Gra = (0.5 * (aw + np.sqrt(- aw2 * (ha-1))))[0]
    # Gi = 0.5 * (aw + np.sqrt(aw2 * (1-height)))
    # Gr = 0.5 * (aw - np.sqrt(- aw2 * (height-1)))
    print(f"{k_0=}")
    print(f"Gia = {Gia/k_0}")
    print(f"Gra = {Gra/k_0}")

    cb, hb, wb =  get_SPR_params(gap*um,20*um, 60, visual_debug)
    kb = n * k_0 * np.sin(np.pi/180*cb)
    print(f"{k_0=}")
    print(f"ka = {ka/k_0}")
    print(f"kb = {kb/k_0}")

    aw = a * wb
    aw2 = aw*aw
    Gib = (0.5 * (aw - np.sqrt(aw2 * (1-hb))))[0]
    Grb = (0.5 * (aw + np.sqrt(- aw2 * (hb-1))))[0]
    # Gi = 0.5 * (aw + np.sqrt(aw2 * (1-height)))
    # Gr = 0.5 * (aw - np.sqrt(- aw2 * (height-1)))
    print(f"Gib = {Gib/k_0}")
    print(f"Grb= {Grb/k_0}")
    
    prism_setup = ExperimentSPR()
    prism_setup.wavelength = 197*um
    prism_setup.add(Layer(1.53,1))
    prism_setup.add(Layer(1,1))
    r_21 = prism_setup.R(angles = [ca])[0]
    print(f"{r_21=}")

    def signal(k_x, r_21 = 0.95, w_0 = 500 * k_0, x_0 = 250 * k_0, 
               k_xrma = 1.05 * k_0, k_xrmb = 1.03 * k_0,
               G_ta = 0.005 * k_0, G_ra = 0.005 * k_0 / 2, G_tb = 0.005 * k_0, G_rb = 0.005 * k_0 / 2):
        pos_x = x_range[x_range >= 0]
        non_neg_x = x_range[x_range < 0]
        E_0 = 1
        E = lambda x: E_0 * np.exp(- (x-x_0)**2 / (w_0*w_0))
        # k_x = 1.03 * k_0
        D_0a = lambda x: -2 * G_ra * E(x) * r_21 /(G_ta + 1.0j * (k_x - k_xrma))
        D_0b = lambda x: -2 * G_rb * E(x) * r_21 /(G_tb + 1.0j * (k_x - k_xrmb))
        # k_zb = np.sqrt()
        # k_za**2  + k_xa**2 = eps_a*k_0**2
        # t_sp = np.sqrt(k_xrma/k_xrma)* 2 * k_zb / (k_za + k_zb)
        t_sp = 1
        Dplus = lambda x: D_0b(x) + (t_sp * D_0a(x) - D_0b(x)) * np.exp(-(G_tb + 1.0j*(k_x - k_xrmb))*x)
        Dminus = lambda x: D_0a(x)  #+ ((1-t_sp) * D_0b(x) - D_0a(x)) * np.exp((G_t + 1.0j*(k_x - k_xrma))*x)
        Reflected_pos = lambda x: E(x) * r_21 + Dplus(x)
        Reflected_neg = lambda x: E(x) * r_21  + Dminus(x)
        y_pos = np.abs(Reflected_pos(pos_x))
        y_neg = np.abs(Reflected_neg(non_neg_x))
        return np.concatenate((y_neg, y_pos), axis=None)
    
    
    
    r = signal(n * k_0 * np.sin(np.pi * 42.1 / 180),r_21 = r_21, w_0 = w0, x_0 = x_0,
               k_xrma= ka, k_xrmb= kb,
               G_ta = Gia+Gra, G_ra = Gra, G_tb = Gib+Grb, G_rb = Grb)
    
    return r
       

def main():
    x_range = np.linspace(-0.2,0.2,20001)
    w0 = 0.15
    r = []
    gaps = [100,150, 200]
    for g in gaps:
        r.append(yeatman(x_range,w0, g, 0)**2)
        
    def gaussuan(x_range, w_0 = w0, x_0 = 0, E_0 = 1):
        E = lambda x: E_0 * np.exp(- (x-x_0)**2 / (w_0*w_0))
        return np.abs(E(x_range))**2
    
    styles = ['-.','--','-']
    fig, ax = plt.subplots(1, 1, dpi=300)
    ax.plot(x_range*100, gaussuan(x_range),label="beam", linestyle='-', color='r')
    for i in range(len(gaps)):
        ax.plot(x_range*100, r[i], label=f"{gaps[i]}", linestyle=styles[i])
    
    
    
    plt.ylabel("Reflectance $R_p$")
    plt.xlabel("$\mathit{x}$ (mm)")
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title="$\delta$ (μm)")
    plt.savefig("Figure_1.png", dpi=600)
    plt.show() 

if __name__ == '__main__':
    main()