# local imports
import sys
#sys.path.append('E:\Python\Common\SPPPy')
#sys.path.append('E:\Python\Common\common')

from SPPPy import ExperimentSPR, Layer, MaterialDispersion
from common.optics import *
# from helper import *
# from settings import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import sys

# general imports
from pathlib import Path

import numpy as np
from numpy.lib import scimath as SM
import matplotlib.pyplot as plt

def reflectivity(theta):
    n0 = 1.65
    eps0 = np.power(n0,2)
    k_z0 = SM.sqrt(eps0 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    k_z1 = SM.sqrt(eps1 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    k_z2 = SM.sqrt(eps2 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    k_z3 = SM.sqrt(eps3 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    r01 = (k_z0 * eps1 - k_z1 * eps0) / (k_z0 * eps1 + k_z1 * eps0)
    r12 = (k_z1 * eps2 - k_z2 * eps1) / (k_z1 * eps2 + k_z2 * eps1)
    r23 = (k_z2 * eps3 - k_z3 * eps2) / (k_z2 * eps3 + k_z3 * eps2)
    R = np.power(np.abs((r01 + r12 * np.exp(2.0* 1.0j * k_z1 * d1) + r23 * np.exp(2.0 * 1.0j * (k_z1 * d1 + k_z2 * d2)) + r01 * r12 * r23 * np.exp(2.0* 1.0j * k_z2 * d2))/(1.0 + r01 * r12 * np.exp(2.0* 1.0j * k_z1 * d1) + r01 * r23 * np.exp(2.0 * 1.0j * (k_z1 * d1 + k_z2 * d2)) +  r12 * r23 * np.exp(2.0* 1.0j * k_z2 * d2))),2)
    return  R

def spp_step_diffraction():
    # Рассчитываем потери G
    # Задаём внешнее поле E(x)
    # считаем k_xr и k_x t_sp
    # считаем D+(x) и складываем с E
    # считаем интенсивность отраженного сигнала и строим график
    
    # Запиcываем оптические константы слоёв eps
    n0 = 1.65
    eps0 = np.power(n0,2)
    eps1 = 1
    eps2 = 1
    eps3 = 0.04 - 4.0j # InSb
    
    theta = 45
    
    # Считаем волновые вектора слоёв
    l_speed = 3.0e08
    omega = 2.0 * np.pi * l_speed / wavelength
    kx_0 = omega / l_speed * np.sqrt( eps1 * eps2 / (eps1+eps2))
    def kx(theta,n0):
        return omega / l_speed * n0 * np.sin(theta)
    
    k_z0 = SM.sqrt(eps0 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    k_z1 = SM.sqrt(eps1 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    k_z2 = SM.sqrt(eps2 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))
    k_z3 = SM.sqrt(eps3 * np.power((omega / l_speed),2) - np.power(kx(theta,n0),2))

    # Рассчитываем френелевские коэффциенты и строим резонансную кривую
    r01 = (k_z0 * eps1 - k_z1 * eps0) / (k_z0 * eps1 + k_z1 * eps0)
    r12 = (k_z1 * eps2 - k_z2 * eps1) / (k_z1 * eps2 + k_z2 * eps1)
    r23 = (k_z2 * eps3 - k_z3 * eps2) / (k_z2 * eps3 + k_z3 * eps2)
    
    R = np.power(np.abs((r01 + r12 * np.exp(2.0* 1.0j * k_z1 * d1) + r23 * np.exp(2.0 * 1.0j * (k_z1 * d1 + k_z2 * d2)) + r01 * r12 * r23 * np.exp(2.0* 1.0j * k_z2 * d2))/(1.0 + r01 * r12 * np.exp(2.0* 1.0j * k_z1 * d1) + r01 * r23 * np.exp(2.0 * 1.0j * (k_z1 * d1 + k_z2 * d2)) +  r12 * r23 * np.exp(2.0* 1.0j * k_z2 * d2))),2)

    # Находим резонансный угол и резонансный минимумы при данной величине зазора
    
    G_t = G_r + G_i
    
    R = 1 - 4 * G_i * G_i / G_t**2
    
    
    
    D0 = ( - 2 * G_r * E(x) * r_21 )/( G_t + 1.0j  * (k_x - k_xrm) )


def YeatmanDynamicModel():
    # 0 - prism
    # 1 - металл
    # 2 - призма
    # 3 - тонкая плёнка
       
    x_range = np.linspace(-1000,1000,2001)
    k_0 = 1
    
    def signal(k_x, r_21 = 0.95, w_0 = 500 * k_0, x_0 = 250 * k_0, 
               k_xrma = 1.05 * k_0, k_xrmb = 1.03 * k_0,
               G_t = 0.005 * k_0, G_r = 0.005 * k_0 / 2):
        pos_x = x_range[x_range >= 0]
        non_neg_x = x_range[x_range < 0]
        E_0 = 1
        E = lambda x: E_0 * np.exp(- (x-x_0)**2 / (w_0*w_0))
        # k_x = 1.03 * k_0
        D_0a = lambda x: -2 * G_r * E(x) * r_21 /(G_t + 1.0j * (k_x - k_xrma))
        D_0b = lambda x: -2 * G_r * E(x) * r_21 /(G_t + 1.0j * (k_x - k_xrmb))
        # k_zb = np.sqrt()
        # k_za**2  + k_xa**2 = eps_a*k_0**2
        # t_sp = np.sqrt(k_xrma/k_xrma)* 2 * k_zb / (k_za + k_zb)
        t_sp = 0.9
        Dplus = lambda x: D_0b(x) + (t_sp * D_0a(x) - D_0b(x)) * np.exp(-(G_t + 1.0j*(k_x - k_xrmb))*x)
        Dminus = lambda x: D_0a(x)  #+ ((1-t_sp) * D_0b(x) - D_0a(x)) * np.exp((G_t + 1.0j*(k_x - k_xrma))*x)
        Reflected_pos = lambda x: E(x) * r_21 + Dplus(x)
        Reflected_neg = lambda x: E(x) * r_21  + Dminus(x)
        y_pos = np.abs(Reflected_pos(pos_x))
        y_neg = np.abs(Reflected_neg(non_neg_x))
        return np.concatenate((y_neg, y_pos), axis=None)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.3])
    ax.set_ylabel('$R_p$')
    ax.set_xlabel('x')
    
    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(left=0.25, bottom=0.25)
    title = """The model reproduced from Yeatman (1996) 
doi:10.1016/0956-5663(96)83298-2
altered with Gaussian light"""
    fig.suptitle(title, fontsize=10)
    
    k_x0 = 1.03 * k_0
    [line] = ax.plot(x_range, signal(k_x0), linewidth=2, color='red')
    
    # Define an axes area and draw a slider in it
    axis_color = 'lightgoldenrodyellow'
    amp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
    amp_slider = Slider(amp_slider_ax, '$k_x$', 0.9*k_0, 1.1*k_0, valinit=k_x0)
    
    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        line.set_ydata(signal(amp_slider.val))
        fig.canvas.draw_idle()
    amp_slider.on_changed(sliders_on_changed)
    
    # Add a button for resetting the parameters
    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
    def reset_button_on_clicked(mouse_event):
        amp_slider.reset()
    reset_button.on_clicked(reset_button_on_clicked)

    plt.show()

um = 1e-6

def critical_angle():
    structure = ExperimentSPR()
    structure.wavelength = 197*um
    structure.add(Layer(1.5, 1))
    structure.add(Layer(1, 200*um))
    structure.add(Layer(eps2nk(-19.16 + 2.8j), 1))

    

    

    

    


