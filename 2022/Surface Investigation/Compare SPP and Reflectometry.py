# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:31:43 2021

@author: Leon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

import seaborn as sns
sns.set()
sns.set_theme(style="whitegrid", palette="dark")

from numpy.lib import scimath as SM

from Helper import get_eps, open_optical_data, get_n_glass, radian,get_pyrex_n, reflectivity, DEFAULT_WAVELENGTH

DEFAULT_WAVELENGTH = 632.8e-9
WAVELENGTHS = [632.8e-9]
THETA = np.linspace(55,85,60)
# 20 nm-thick germanium film deposited directly on SiO2 substrate
Au_n, Au_k = open_optical_data(r"Olmon 2012 Au-evaporated.csv", 451)
# Oxide films were deposited by reactive electron-beam evaporation onto various sorts of substrates at 300 °C
SiO2_n, SiO2_k = open_optical_data(r"SiO2 thin film Rodriguez-de Marcos.csv", 398)
Ge_n, Ge_k = open_optical_data(r"Ge Ciesielski-20nm.csv",281)
Au_eps = get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k)

METALS = [get_eps(wl*1e6, Au_n,Au_k) for wl in WAVELENGTHS]
PRISMS = [get_n_glass(wl*1e6) for wl in WAVELENGTHS]

METAL_THICKNESS = 45e-9
GRADIENT_THICKNESS = 20e-9
GRADIENT_N_PROFILE = lambda x: 1.5 + 0.001*np.sin(x)

NUMBER_OF_LAYERS = 4
LAYERS_QUILITY = 100

LAYER_THICKNESS_STEP = 1e-9

NUMBER_OF_MEASURMENTS = 100

# график разницы тонкой плёнки
theta = np.linspace(55,65,NUMBER_OF_MEASURMENTS)
RthetaN0 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]
RthetaN1 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),get_eps(DEFAULT_WAVELENGTH*1e6, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP*1 ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]


plt.title("Сравнение отклонений при угловом сканировании")
plt.plot(theta, RthetaN0, '--', label='z=0 нм')
plt.plot(theta,RthetaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('θ, °')
plt.legend()
plt.show()

WL = np.linspace(500,700,NUMBER_OF_MEASURMENTS)
RlambdaN0 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl*1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP*0 ,0], radian(63),wl*1e-9) for wl in WL]
RlambdaN1 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl*1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP*1 ,0], radian(63),wl*1e-9) for wl in WL]

plt.title("Сравнение отклонений ППР \n при сканировании на длине волны")
plt.plot(WL, RlambdaN0, '--', label='z=0 нм')
plt.plot(WL,RlambdaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()

REFLlambdaN = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(0),wl*1e-9) for wl in WL]
REFLlambdaN1 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl*1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP ,0], radian(0),wl*1e-9) for wl in WL]

plt.title("Сравнение отклонений рефлектометрия \n при сканировании на длине волны")
plt.plot(WL, REFLlambdaN, '--', label='z=0 нм')
plt.plot(WL,REFLlambdaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()

def deviation(N,N1):
    return np.max(np.abs(np.array(N)-np.array(N1)))
def percent(N):
    return np.array(N)*100

z = [LAYER_THICKNESS_STEP*i for i in range(50)]
RthetaNs = []
for zi in z:
    RthetaNs.append([reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),get_eps(DEFAULT_WAVELENGTH *1e6, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, zi ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta])
dRthetaNs = []
for i in range(len(RthetaNs)-1):
    dRthetaNs.append(deviation(RthetaNs[i],RthetaNs[i+1]))
    
# plt.title("Сравнение отклонений ППР при угловом сканировании")
plt.plot(np.array(z[1:])*1e9,percent(dRthetaNs), '+', label='θ ППР (λ = '+str(DEFAULT_WAVELENGTH*1e9)+' нм)')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
# plt.show()

THETA_SCAN = 58
RlambdaNs = []
for zi in z:  
    RlambdaNs.append([reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl *1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, zi ,0], radian(THETA_SCAN),wl*1e-9) for wl in WL])
dRlambdaNs = []
for i in range(len(RlambdaNs)-1):
    dRlambdaNs.append(deviation(RlambdaNs[i],RlambdaNs[i+1]))

# plt.title("Сравнение отклонений ППР при спектральном сканировании")
plt.plot(np.array(z[1:])*1e9,percent(dRlambdaNs), 'x', label='λ ППР (θ = '+str(THETA_SCAN)+'°)')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
# plt.show()



REFLlambdaNs = []
for zi in z:  
    REFLlambdaNs.append([reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl *1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, zi ,0], radian(0),wl*1e-9) for wl in WL])
dREFLlambdaNs = []
for i in range(len(REFLlambdaNs)-1):
    dREFLlambdaNs.append(deviation(REFLlambdaNs[i],REFLlambdaNs[i+1]))

# plt.title("Сравнение отклонений рефлектометрии \n при спектральном сканировании")
# plt.title("Сравнение чувствительности методов")
plt.plot(np.array(z[1:])*1e9,percent(dREFLlambdaNs),'.' , label='λ рефлектометрия')
plt.ylabel('max(|ΔR|), %')
plt.xlabel('z, нм')
plt.legend()
plt.show()

import seaborn as sns
sns.set()
sns.set_theme(style="whitegrid", palette="dark")


DEFAULT_WAVELENGTH = 632.8e-9
WAVELENGTHS = [632.8e-9]
THETA = np.linspace(55,75,60)
# 20 nm-thick germanium film deposited directly on SiO2 substrate
Au_n, Au_k = open_optical_data(r"Olmon 2012 Au-evaporated.csv", 451)
# Oxide films were deposited by reactive electron-beam evaporation onto various sorts of substrates at 300 °C
SiO2_n, SiO2_k = open_optical_data(r"SiO2 thin film Rodriguez-de Marcos.csv", 398)
Ge_n, Ge_k = open_optical_data(r"Ge Ciesielski-20nm.csv",281)
Au_eps = get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k)

METALS = [get_eps(wl*1e6, Au_n,Au_k) for wl in WAVELENGTHS]
PRISMS = [get_n_glass(wl*1e6) for wl in WAVELENGTHS]

METAL_THICKNESS = 45e-9
GRADIENT_THICKNESS = 10e-9
GRADIENT_N_PROFILE = lambda x: 1.5 + 0.001*np.sin(x)

NUMBER_OF_LAYERS = 4
LAYERS_QUILITY = 100

LAYER_THICKNESS_STEP = 1e-9

NUMBER_OF_MEASURMENTS = 100

nxdepth = np.linspace(0.05,1,20)


def parametrize_curve(func,d):
    points = np.linspace(0,1,LAYERS_QUILITY, endpoint=False)
    epsilon = [func(x)*func(x) for x in points]
    d = [d/len(points)]*len(points)
    return epsilon,d
# '''
WL = np.linspace(500,700,NUMBER_OF_MEASURMENTS)

normal_effective_depth = 0.05
ns = get_n_glass(DEFAULT_WAVELENGTH*1e6)
nb = get_pyrex_n(DEFAULT_WAVELENGTH*1e9)
# ns = get_pyrex_n
# diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
# diffusion = lambda x: 1.5 + 0.1*np.sin(10*x)
diffusion = lambda x: nb + 0.5*(ns -nb)* sp.special.erfc((2*x-1)/normal_effective_depth)
x = np.linspace(0,1,1000)
plt.plot(x, diffusion(x))
plt.show()

'''
epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
depth = []
depth.append(0)
for i in range(1,len(d)):
    depth.append((depth[i-1]+d[i]*1e9))
plt.plot(depth, SM.sqrt(np.array(epsilon)))
plt.title("index profile")
plt.yticks([1.4,1.42,1.44,1.46,1.48,1.5])
plt.xticks([0,5,10,15,20])
plt.show()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# график различия для тонкой плёнки
theta = np.linspace(57,64,NUMBER_OF_MEASURMENTS)

RthetaN0 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),ns**2,nb**2,1],[0,METAL_THICKNESS,GRADIENT_THICKNESS/2,GRADIENT_THICKNESS/2,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]
RthetaN1 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS] + d + [0], radian(th),DEFAULT_WAVELENGTH) for th in theta]


plt.title("Сравнение отклонений при угловом сканировании")
plt.plot(theta, RthetaN0, '--', label='z=0 нм')
plt.plot(theta,RthetaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('θ, °')
plt.legend()
plt.show()

def deviation(N,N1):
    return np.max(np.abs(np.array(N)-np.array(N1)))


print(deviation(RthetaN0,RthetaN1), 'theta' )


dRWLdepth = []
dREFLWLdepth = []


for xdepth in nxdepth:
    dRWL = []
    for th in theta:
        # RlambdaN0 = []
        RlambdaN1 = []
        WL = np.linspace(500,700,NUMBER_OF_MEASURMENTS)
        for i in range(len(WL)):
            wl = WL[i]
            normal_effective_depth = xdepth
            ns = get_n_glass(wl*1e-3)
            nb = get_pyrex_n(wl)
            # ns = get_pyrex_n
            # diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
            # diffusion = lambda x: 1.5 + 0.1*np.sin(10*x)
            diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
            x = np.linspace(0,1,1000)
            # plt.plot(x, diffusion(x))
            # plt.show()
            
            epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
            
            # RlambdaN0.append(reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),ns**2,nb**2,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS/2,GRADIENT_THICKNESS/2 ,0], radian(th),wl*1e-9))
            RlambdaN1.append(reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS] + d + [0], radian(th),wl*1e-9))
            
            # normal_effective_depth = 0.1
            # epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
            # RlambdaN0.append(reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS] + d + [0], radian(th),wl*1e-9))
        
        # plt.title("Сравнение отклонений ППР \n при сканировании на длине волны θ = " +str(th) +"°")
        # plt.plot(WL, RlambdaN0, '--', label='z=0 нм')
        # plt.plot(WL,RlambdaN1, label='Δz=1 нм')
        # plt.ylabel('ΔR')
        # plt.xlabel('λ, нм')
        # plt.legend()
        # plt.show()
        # dv = deviation(RlambdaN0,RlambdaN1)
        dRWL.extend(RlambdaN1)
        # print(dv, " : ", th)
    dRWLdepth.append(dRWL)
        
    REFLlambdaN1 = []
    for i in range(len(WL)):
        wl = WL[i]
        normal_effective_depth = xdepth
        ns = get_n_glass(wl*1e-3)
        nb = get_pyrex_n(wl)
        # ns = get_pyrex_n
        # diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
        # diffusion = lambda x: 1.5 + 0.1*np.sin(10*x)
        diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
        x = np.linspace(0,1,1000)
        # plt.plot(x, diffusion(x))
        # plt.show()
        
        epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
        
        # REFLlambdaN0.append(reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),ns**2,nb**2,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS/2,GRADIENT_THICKNESS/2 ,0], radian(63),wl*1e-9))
        REFLlambdaN1.append(reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS] + d + [0], radian(0),wl*1e-9))
        normal_effective_depth = 0.1
        # epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
        # REFLlambdaN0.append(reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS] + d + [0], radian(0),wl*1e-9))
        # deviation(REFLlambdaN0,REFLlambdaN1)
    
    dREFLWLdepth.append(REFLlambdaN1)
    print(xdepth)

xdRWLdepth = []
for i in range(len(dRWLdepth)-1):
    xdRWLdepth.append(deviation(dRWLdepth[i],dRWLdepth[i+1]))
print('R: ', xdRWLdepth)
    
xdREFLWLdepth = []
for i in range(len(dREFLWLdepth)-1):
    xdREFLWLdepth.append(deviation(dREFLWLdepth[i],dREFLWLdepth[i+1]))
print('REFL: ', xdREFLWLdepth)
'''
xdREFLWLdepth = [3.91062755400462e-07, 4.893279613171764e-07, 6.013995801357197e-07, 7.274347050834074e-07, 8.575417977385591e-07, 9.874184347946624e-07, 1.115920833849593e-06, 1.2388109363881128e-06, 1.3485086781672706e-06, 1.4371170408700173e-06, 1.4996360171437573e-06, 1.5350003204628315e-06, 1.5453030724477124e-06, 1.5344249784665642e-06, 1.5068488206648745e-06, 1.4668958038255298e-06, 1.4183389117627954e-06, 1.364270769910192e-06, 1.3071150771759577e-06]
xdRWLdepth = [3.294329325492784e-05, 3.302941785393587e-05, 3.31150468816932e-05, 3.320016820357319e-05, 3.328333535818073e-05, 3.333668953542368e-05, 3.321836632830255e-05, 3.265458298706836e-05, 3.1416411131357513e-05, 2.947370469519317e-05, 2.697665368006641e-05, 2.4152352625539386e-05, 2.1218750795415353e-05, 1.8343996126934936e-05, 1.5639061367644924e-05, 1.3166200865943889e-05, 1.0951855853369441e-05, 8.99866481007816e-06, 7.294825877579569e-06]

# plt.title("Сравнение чувствительности методов")
plt.plot(np.array(nxdepth[:-1])*GRADIENT_THICKNESS*1e9, np.array(xdRWLdepth)*1000, '+' , label='ППР')
plt.plot(np.array(nxdepth[:-1])*GRADIENT_THICKNESS*1e9, np.array(xdREFLWLdepth)*1000, '.', label='Рефлектометрия')
plt.ylabel('max(|ΔR|), ‰')
plt.xlabel('d, нм')
plt.legend()
plt.show()

# plt.title("Сравнение отклонений Рефл. \n при сканировании на длине волны")
# plt.plot(WL, REFLlambdaN0, '--', label='z=0 нм')
# plt.plot(WL,REFLlambdaN1, label='Δz=1 нм')
# plt.ylabel('ΔR')
# plt.xlabel('λ, нм')
# plt.legend()
# plt.show()

# print('REFL: ', deviation(REFLlambdaN0,REFLlambdaN1))


'''
# график разницы тонкой плёнки
theta = np.linspace(55,65,NUMBER_OF_MEASURMENTS)
RthetaN0 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]
RthetaN1 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),get_eps(DEFAULT_WAVELENGTH*1e6, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP*1 ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]


plt.title("Сравнение отклонений при угловом сканировании")
plt.plot(theta, RthetaN0, '--', label='z=0 нм')
plt.plot(theta,RthetaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('θ, °')
plt.legend()
plt.show()

WL = np.linspace(500,700,NUMBER_OF_MEASURMENTS)
RlambdaN0 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl*1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP*0 ,0], radian(63),wl*1e-9) for wl in WL]
RlambdaN1 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl*1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP*1 ,0], radian(63),wl*1e-9) for wl in WL]

plt.title("Сравнение отклонений ППР \n при сканировании на длине волны")
plt.plot(WL, RlambdaN0, '--', label='z=0 нм')
plt.plot(WL,RlambdaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()

REFLlambdaN = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(0),wl*1e-9) for wl in WL]
REFLlambdaN1 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl*1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP ,0], radian(0),wl*1e-9) for wl in WL]

plt.title("Сравнение отклонений рефлектометрия \n при сканировании на длине волны")
plt.plot(WL, REFLlambdaN, '--', label='z=0 нм')
plt.plot(WL,REFLlambdaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()

def deviation(N,N1):
    return np.sum(np.abs(np.array(N)-np.array(N1)))/NUMBER_OF_MEASURMENTS
def percent(N):
    return np.array(N)*100

z = [LAYER_THICKNESS_STEP*i for i in range(50)]
RthetaNs = []
for zi in z:
    RthetaNs.append([reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),get_eps(DEFAULT_WAVELENGTH *1e6, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, zi ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta])
dRthetaNs = []
for i in range(len(RthetaNs)-1):
    dRthetaNs.append(deviation(RthetaNs[i],RthetaNs[i+1]))
    
plt.title("Сравнение отклонений ППР при угловом сканировании")
plt.plot(np.array(z[0:-1])*1e9,percent(dRthetaNs), label='θ ППР (λ = '+str(DEFAULT_WAVELENGTH*1e9)+' нм)')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
# plt.show()

THETA_SCAN = 58
RlambdaNs = []
for zi in z:  
    RlambdaNs.append([reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl *1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, zi ,0], radian(THETA_SCAN),wl*1e-9) for wl in WL])
dRlambdaNs = []
for i in range(len(RlambdaNs)-1):
    dRlambdaNs.append(deviation(RlambdaNs[i],RlambdaNs[i+1]))

plt.title("Сравнение отклонений ППР при спектральном сканировании")
plt.plot(np.array(z[0:-1])*1e9,percent(dRlambdaNs), '--', label='λ ППР (θ = '+str(THETA_SCAN)+'°)')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
# plt.show()



REFLlambdaNs = []
for zi in z:  
    REFLlambdaNs.append([reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),get_eps(wl *1e-3, SiO2_n,SiO2_k),1],[0,METAL_THICKNESS, zi ,0], radian(0),wl*1e-9) for wl in WL])
dREFLlambdaNs = []
for i in range(len(REFLlambdaNs)-1):
    dREFLlambdaNs.append(deviation(REFLlambdaNs[i],REFLlambdaNs[i+1]))

plt.title("Сравнение отклонений рефлектометрии \n при спектральном сканировании")
plt.title("Сравнение чувствительности методов")
plt.plot(np.array(z[0:-1])*1e9,percent(dREFLlambdaNs),'.' , label='λ рефлектометрия')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
plt.legend()
plt.show()


# график разницы тонкой плёнки
theta = np.linspace(55,65,NUMBER_OF_MEASURMENTS)
ns = get_n_glass(DEFAULT_WAVELENGTH*1e6)
RthetaN = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]
RthetaN1 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]

RthetaN = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]
RthetaN1 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]


plt.title("Сравнение отклонений при угловом сканировании")
plt.plot(theta, RthetaN, '--', label='z=0 нм')
plt.plot(theta,RthetaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('θ, °')
plt.legend()
plt.show()

WL = np.linspace(500,700,NUMBER_OF_MEASURMENTS)
RlambdaN = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(63),wl*1e-9) for wl in WL]
RlambdaN1 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP ,0], radian(63),wl*1e-9) for wl in WL]

plt.title("Сравнение отклонений ППР \n при сканировании на длине волны")
plt.plot(WL, RlambdaN, '--', label='z=0 нм')
plt.plot(WL,RlambdaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()

REFLlambdaN = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),1],[0,METAL_THICKNESS,0], radian(0),wl*1e-9) for wl in WL]
REFLlambdaN1 = [reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, LAYER_THICKNESS_STEP ,0], radian(0),wl*1e-9) for wl in WL]

plt.title("Сравнение отклонений рефлектометрии \n при сканировании на длине волны")
plt.plot(WL, REFLlambdaN, '--', label='z=0 нм')
plt.plot(WL,REFLlambdaN1, label='Δz=1 нм')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()

def deviation(N,N1):
    return np.sum(np.abs(np.array(N)-np.array(N1)))/NUMBER_OF_MEASURMENTS
def percent(N):
    return np.array(N)*100

z = [LAYER_THICKNESS_STEP*i for i in range(50)]
RthetaNs = []
for zi in z:
    RthetaNs.append([reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, zi ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta])
dRthetaNs = []
for i in range(len(RthetaNs)-1):
    dRthetaNs.append(deviation(RthetaNs[i],RthetaNs[i+1]))
    
plt.title("Сравнение отклонений ППР при угловом сканировании")
plt.plot(np.array(z[0:-1])*1e9,percent(dRthetaNs), label='θ ППР')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
# plt.show()

RlambdaNs = []
for zi in z:  
    RlambdaNs.append([reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, zi ,0], radian(63),wl*1e-9) for wl in WL])
dRlambdaNs = []
for i in range(len(RlambdaNs)-1):
    dRlambdaNs.append(deviation(RlambdaNs[i],RlambdaNs[i+1]))

plt.title("Сравнение отклонений ППР при спектральном сканировании")
plt.plot(np.array(z[0:-1])*1e9,percent(dRlambdaNs), '--', label='λ ППР')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
# plt.show()



REFLlambdaNs = []
for zi in z:  
    REFLlambdaNs.append([reflectivity([get_n_glass(wl*1e-3), get_eps(wl*1e-3, Au_n,Au_k),(ns)**2,1],[0,METAL_THICKNESS, zi ,0], radian(0),wl*1e-9) for wl in WL])
dREFLlambdaNs = []
for i in range(len(REFLlambdaNs)-1):
    dREFLlambdaNs.append(deviation(REFLlambdaNs[i],REFLlambdaNs[i+1]))

plt.title("Сравнение отклонений рефлектометрии \n при спектральном сканировании")
plt.title("Сравнение чувствительности методов")
plt.plot(np.array(z[0:-1])*1e9,percent(dREFLlambdaNs),'.' , label='λ рефлектометрия')
plt.ylabel('ΔR, %')
plt.xlabel('z, нм')
plt.legend()
plt.show()

# nb = get_pyrex_n(DEFAULT_WAVELENGTH*1e9)
# diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
# epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
# RWL = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(th), DEFAULT_WAVELENGTH) for th in theta]
# RWL2 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),(ns+nb)**2/4,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]


#'''
'''







def parametrize_curve(func,d):
    points = np.linspace(0,1,LAYERS_QUILITY, endpoint=False)
    epsilon = [func(x)*func(x) for x in points]
    d = [d/len(points)]*len(points)
    return epsilon,d





normal_effective_depth = 0.9
ns = 1.5
nb = 1.4
# ns = get_pyrex_n
# diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
# diffusion = lambda x: 1.5 + 0.1*np.sin(10*x)
diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
x = np.linspace(0,1,100)
# plt.plot(x, diffusion(x))
# plt.show()

epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
depth = []
depth.append(0)
for i in range(1,len(d)):
    depth.append((depth[i-1]+d[i]*1e9))
plt.plot(depth, SM.sqrt(np.array(epsilon)))
plt.title("index profile")
plt.yticks([1.4,1.42,1.44,1.46,1.48,1.5])
plt.xticks([0,5,10,15,20])
plt.show()

THEORETICAL_REFLETOMETRY = []
THEORETICAL_REFLETOMETRY2 = []

WL = np.linspace(500,700,100)
for i, wli in enumerate(WL):
    ns = get_n_glass(wli*1e-3)
    nb = get_pyrex_n(wli)
    diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
    epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
    THEORETICAL_REFLETOMETRY.append(reflectivity([get_n_glass(wli*1e-3), get_eps(wli*1e-3, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(0), wli*1e-9))
    THEORETICAL_REFLETOMETRY2.append(reflectivity([get_n_glass(wli*1e-3), get_eps(wli*1e-3, Au_n,Au_k),(ns+nb)**2/4,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(0),wli*1e-9))

plt.plot(WL, THEORETICAL_REFLETOMETRY)
plt.plot(WL, THEORETICAL_REFLETOMETRY2,'+')
# plt.yscale('log')
plt.title("Reflectometry")
plt.show()



delta1 = [xi*xi for xi in (np.array(THEORETICAL_REFLETOMETRY)-np.array(THEORETICAL_REFLETOMETRY2))]
max_delta1 = max(delta1)
print('max R: ', max(delta1))
print('sum R: ', np.sum(np.power(max_delta1,1)))



plt.plot(theta, RWL)
plt.plot(theta, RWL2,'+')
plt.title("SPR Reflectometry")
plt.show()

delta2 = [xi*xi for xi in (np.array(RWL)-np.array(RWL2))]
max_delta2 = max(delta2)
print('max SPR: ', max_delta2)
print('theta: ', theta[delta2.index(max_delta2)])
print('effective SPR\R: ',max_delta2/max_delta1)
print('sum SPR: ', np.sum(np.power(max_delta2,1)))

SPP_SPRECTROSCOPY = []
SPP_SPRECTROSCOPY2 = []


for i, wli in enumerate(WL):
    ns = get_n_glass(wli*1e-3)
    nb = get_pyrex_n(wli)
    diffusion = lambda x: nb - 0.5*(nb-ns)* sp.special.erfc((2*x-1)/normal_effective_depth)
    epsilon,d = parametrize_curve(diffusion, GRADIENT_THICKNESS)
    SPP_SPRECTROSCOPY.append(reflectivity([get_n_glass(wli*1e-3), get_eps(wli*1e-3, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(63), wli*1e-9))
    SPP_SPRECTROSCOPY2.append(reflectivity([get_n_glass(wli*1e-3), get_eps(wli*1e-3, Au_n,Au_k),(ns+nb)**2/4,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(63),wli*1e-9))

plt.plot(WL, SPP_SPRECTROSCOPY)
plt.plot(WL, SPP_SPRECTROSCOPY2,'+')
# plt.yscale('log')
plt.title("SPP Spectroreflectometry")
plt.show()

delta3 = [xi*xi for xi in (np.array(SPP_SPRECTROSCOPY)-np.array(SPP_SPRECTROSCOPY2))]
max_delta3 = max(delta3)
print('max R: ', max(delta3))
print('sum R: ', np.sum(np.power(max_delta3,1)))
print('effective SSPR\R: ',max_delta3/max_delta1)

plt.title("Сравнение отклонений на углах θ 0° и 63°")
plt.plot(WL,np.array(THEORETICAL_REFLETOMETRY)-np.array(THEORETICAL_REFLETOMETRY2), '--', label='Рефлектометрия')
plt.plot(WL,np.array(SPP_SPRECTROSCOPY)-np.array(SPP_SPRECTROSCOPY2), label='ППР')
plt.ylabel('ΔR')
plt.xlabel('λ, нм')
plt.legend()
plt.show()



# epsilon,d = parametrize_curve(GRADIENT_N_PROFILE, GRADIENT_THICKNESS)
# THEORETICAL_REFLETOMETRY = [reflectivity([get_n_glass(wl*1e-6), get_eps(wl*1e-6, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(0),wl*1e-9) for wl in WL]
# THEORETICAL_REFLETOMETRY2 = [reflectivity([get_n_glass(wl*1e-6), get_eps(wl*1e-6, Au_n,Au_k),2.25,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(0),wl*1e-9) for wl in WL]




# theta = np.linspace(40,80,100)
# epsilon,d = parametrize_curve(GRADIENT_N_PROFILE, GRADIENT_THICKNESS)
# RWL = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(th), DEFAULT_WAVELENGTH) for th in theta]
# RWL2 = [reflectivity([get_n_glass(DEFAULT_WAVELENGTH*1e6), get_eps(DEFAULT_WAVELENGTH*1e6, Au_n,Au_k),2.25,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(th),DEFAULT_WAVELENGTH) for th in theta]

# delta2 = [xi*xi for xi in (np.array(RWL)-np.array(RWL2))]
# max_delta2 = max(delta2)
# print(max(delta2))
# print(theta[delta2.index(max(delta2))])
# print(max_delta2/max_delta1)

'''
# THEORETICAL_RWL2=[]
# for i in range(len(WAVELENGTHS)):
#     epsilon,d = parametrize_curve(GRADIENT_N_PROFILE, GRADIENT_THICKNESS)
#     THEORETICAL_RWL.append(np.array([reflectivity([PRISMS[i] , METALS[i]]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(th),WAVELENGTHS[i]) for th in THETA]))
#     THEORETICAL_RWL2.append(np.array([reflectivity([PRISMS[i] , METALS[i],2.25,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(th),WAVELENGTHS[i]) for th in THETA]))




# plt.plot(theta,RWL)
# plt.plot(theta,RWL2, '+')
# plt.title("Gradient resonance curve")
# plt.show()


# delta = []
# for th in range(0,90):
#     RWL = [reflectivity([get_n_glass(wl*1e-6), get_eps(wl*1e-6, Au_n,Au_k)]+epsilon+[1],[0,METAL_THICKNESS]+d+[0], radian(th), wl*1e-9) for wl in WL]
#     RWL2 = [reflectivity([get_n_glass(wl*1e-6), get_eps(wl*1e-6, Au_n,Au_k),2.25,1],[0,METAL_THICKNESS, GRADIENT_THICKNESS ,0], radian(th),wl*1e-9) for wl in WL]
#     delta.append(np.sum([xi*xi for xi in (np.array(RWL)-np.array(RWL2))]))
#     print(th)
    # plt.plot(WL,RWL)
    # plt.plot(WL,RWL2, '+')
    # plt.title("Gradient resonance curve")
    # plt.show()

# print(delta.index(min(delta)))

# for i, R in enumerate(THEORETICAL_RWL):
#     plt.plot(THETA,R, label=str(WAVELENGTHS[i]*1e9))
#     plt.plot(THETA,THEORETICAL_RWL2[i], '+', label=str(WAVELENGTHS[i]*1e9))
# plt.title("Gradient resonance curve")
# plt.legend()
# plt.show()

#'''
