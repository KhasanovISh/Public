# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:11:58 2023

@author: Leon
"""

from settings import *
from model import *
from helper import *

import matplotlib.pyplot as plt
import pandas as pd
from labellines import labelLine, labelLines


def plot_set_style(ax):
    plt.rcParams.update({
    "font.family": "monospace"})    
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)
    
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    ax.spines[['right', 'top']].set_visible(False)

def plot_d_theta(ax, gaps, angles):

    ax.plot(angles, np.real(gaps),color='0.6', linestyle = '-', linewidth=4, label="1")
    ax.plot(angles, np.imag(gaps),color='0.6', linestyle = ':', linewidth=4, label="2")

    plt.xlabel(r"$\theta,\rm град$", fontsize = SMALL_SIZE)
    plt.ylabel("$d$, мкм", fontsize = SMALL_SIZE)
    plt.title("(a)")
    
    df = pd.DataFrame(angles, columns = ['Theta град'])
    df['gap мкм'] = gaps

    return df

def plot_minimal_gap(ax, SPR_structure, angles):
    d_min = []
    for th in angles:
        d_min.append(get_minimal_gap(SPR_structure, th))
    ax.plot(angles, d_min, color='0', 
            linestyle='-.', 
            linewidth=1.5, 
            label="3")
    df2 = pd.DataFrame(angles, columns = ['Theta град'])
    df2['gap min мкм'] = d_min
    return df2

def plot_minimal_theta(ax, SPR_structure, gaps):
    th_min = []    
    for d in gaps:
        SPR_structure.layers[1].thickness = d * um
        th_min.append(get_minimal_theta(SPR_structure))
    # (0, (3, 5, 1, 5, 1, 5))
    ax.plot(th_min, gaps, color='0', 
            linestyle='--', 
            linewidth=2, 
            label="4")
    df3 = pd.DataFrame(gaps, columns = ['gap мкм'])
    df3['Theta min град'] = th_min    
    return df3

def save_plot_data(filename, dfs, names = []):
    if len(names) == 0:
        names = list(range(len(dfs)))
    with pd.ExcelWriter(f"{filename}.xlsx") as writer:
        for i, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=names[i])
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    
    
def plot_Agrand(angles, d_range, wavelength, th_crit_pos_y, wl_2_pos):
    FILENAME = f"{wavelength/um:0.0f} um Agrand"

    SPR_structure = setup_SPR_structure(wavelength, 100*um)
    
    gaps = np.array([optimal_gap(SPR_structure, theta) for theta in angles])/um
  
    fig, ax = plt.subplots(1, 1, dpi=300)    

    plot_set_style(ax)
     
  
    df1 = plot_d_theta(ax, gaps, angles)

    tir = np.arcsin(1/3.41)/np.pi*180    
    plt.axvline(x=tir, color='0.6', linestyle=':')
    plt.axhline(y=wavelength/um/2, color='0.6', linestyle=':') 
    plt.axhline(y=0, color='0.8', linestyle='-')
    
    
    plt.text(tir + 0.125, th_crit_pos_y, r"$\rm \theta_{crit}$", rotation = 90, fontsize = SMALL_SIZE+2)
    plt.text(wl_2_pos[0],wl_2_pos[1] , r"$\rm \frac{\lambda}{2}$", fontsize = SMALL_SIZE+2)

    df2 = plot_minimal_gap(ax, SPR_structure, angles)
    df3 = plot_minimal_theta(ax, SPR_structure, d_range)
    
    plt.xlim(np.min(angles), np.max(angles))
    
    labellines_pos = [15,15,15,15]
    labelLines(ax.get_lines(), 
               align=False,  
               xvals=labellines_pos,
               color='k',
               fontsize=SMALL_SIZE-1) 
    
    save_plot_data(FILENAME, [df1,df2,df3], ['Agrand','1','2'])

    plt.show()
    
def plot_SPR(angles, wavelength, d_spp_um, labellines_pos):
    FILENAME = f"{wavelength/um:0.0f} um SPR"

    SPR_structure = setup_SPR_structure(wavelength, 100*um)
    
    fig, ax = plt.subplots(1, 1, dpi=300) 
    
    plot_set_style(ax)

    # ax.plot(X, np.sin(a * X), label=str(a))

    # labelLines(ax.get_lines(), align=False, fontsize=14)
    
    df = pd.DataFrame(angles, columns = ['Theta min град'])
        
    
    LINESTYLES = ['k:', 'k-', 'k+', 'k-.', 'k--','kx-'] 
    for i, k in enumerate([0, 1, 0.25,0.5,1.25,1.5]):
        SPR_structure.layers[1].thickness = d_spp_um * k * um
        rs = SPR_structure.R(angles)
        ax.plot(angles, rs, LINESTYLES[i], label=f"{i}")
        df[f"R_p {i}"] = rs
    labelLines(ax.get_lines(), 
               align=False,  
               xvals=labellines_pos, 
               fontsize=SMALL_SIZE-1)

    tir = np.arcsin(1/3.41)/np.pi*180    
    plt.axvline(x=tir, color='0.6', linestyle=':')
    # plt.axhline(y=wavelength/um/2, color='0.6', linestyle=':') 
    # plt.axhline(y=0, color='0.8', linestyle='-')
    
    
    plt.text(tir + 0.123, 0.85, r"$\rm \theta_{crit}$", rotation = 90, fontsize = SMALL_SIZE+2)
    # plt.text(wl_2_pos[0],wl_2_pos[1] , r"$\rm \frac{\lambda}{2}$", fontsize = SMALL_SIZE+2)


    plt.xlim(np.min(angles), np.max(angles))
    plt.ylim(0,1)
    
    plt.xlabel(r"$\theta,\rm град$", fontsize = SMALL_SIZE)
    plt.ylabel("$R_p$, мкм", fontsize = SMALL_SIZE)
    plt.title("(б)")
    
    save_plot_data(FILENAME, [df], ["SPP"])

    plt.show()