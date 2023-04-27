# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:03:49 2021

@author: Leon
"""

import numpy as np
from scipy import interpolate
from numpy.lib import scimath as SM

LIGHT_SPEED = 3.0e08
LASER_WAVELENGTH = {'HeNe':632.8e-9, 'CO2':10.6e-6, 'FEL':130e-6, 'KTP':532e-9,'Nd:YAG':1064e-9, 'He-Ne 2': 1150e-9, 'He-Ne 3': 3.39e-6}
DEFAULT_WAVELENGTH = LASER_WAVELENGTH['HeNe']

def radian(degree):
    return np.pi * degree / 180

def get_n_ZnSe(x):
    # Amotchkina et al, 2020: n 0.400-13.9 µm, k 0.4-0.888 µm
    return (1-0.689818+4.855169/(1-0.056359/x**2)+0.673922/(1-0.056336/x**2)+2.481890/(1-2222.114/x**2))**.5



def get_n_glass(x):
    #  OHARA - BBH (Barium borate, high-index) L-BBH1
    # return (1+2.56597628/(1-0.0205164012/x**2)+0.526892298/(1-0.0831737967/x**2)+1.27957402/(1-109.78774/x**2))**.5
    # BK7
    return (1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**.5

pyrex_n = [1.479847057488224, 1.4774551671301286, 1.4755484932750693, 1.4740150246864534, 1.4727426082515178, 1.4716194455479255, 1.4705335253390839, 1.4695598291810905, 1.4686604051983427, 1.4678356080812665, 1.4670859343964586]
pyrex_wl = [450, 475.6653992395437, 500.3802281368821, 524.6197718631179, 549.8098859315589, 575, 600.6653992395436, 624.4296577946768, 650.5703422053232, 676.7110266159696, 700]
get_pyrex_n = interpolate.interp1d(pyrex_wl, pyrex_n)


def open_optical_data(filename, strk):
    with open(filename) as f:
        lines = f.readlines()
        Ag_n = np.loadtxt(lines[1:strk-2], delimiter=',')
        Ag_k = np.loadtxt(lines[strk:], delimiter=',')
        return Ag_n, Ag_k


def get_eps(wl, n, k):
    cn = 0
    for i in range(len(n)):
        if wl == n[i,0]:
            cn = n[i,1]
            break
        elif wl < n[i, 0]:
            w1 = n[i-1, 0]
            w2 = n[i, 0]
            n1 = n[i-1, 1]
            n2 = n[i, 1]
            cn = (wl - w1) / (w2 - w1) * (n2 - n1) + n1
            break
    ck = 0
    for i in range(len(k)):
        if wl == k[i,0]:
            ck = k[i,1]
            break
        elif wl < k[i, 0]:
            w1 = k[i-1, 0]
            w2 = k[i, 0]
            k1 = k[i-1, 1]
            k2 = k[i, 1]
            ck = (wl - w1) / (w2 - w1) * (k2 - k1) + k1
            break
    return (cn*cn - ck*ck) + cn * ck * 2.0j

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

def reflectivity(epsilon, d, theta, wavelength=DEFAULT_WAVELENGTH):
    n = [SM.sqrt(ni) for ni in epsilon]
    w = 2.0 * np.pi / wavelength
    a = np.sin(theta)
    a = epsilon[0]*a*a
    k_z = [w* SM.sqrt(epsilon[i] - a) for i in range(1, len(n))]
    k_z.insert(0, w*np.sqrt(epsilon[0] - a))

    r = [(k_z[i]*epsilon[i+1]-k_z[i+1]*epsilon[i]) /
         (k_z[i]*epsilon[i+1]+k_z[i+1]*epsilon[i])
         for i in range(0, len(n)-1)]

    # All layers
    it = 1/(1-r[0])
    M0 = np.array([[it, r[0]*it],
                   [r[0]*it, it]])
    for i in range(1, len(n)-1):
        b = np.exp(-1j*k_z[i]*d[i])
        Mi = np.array([[b/(1-r[i]),
                        b*r[i]/(1-r[i])],
                       [r[i]/(b*(1-r[i])),
                        1/(b*(1-r[i]))]])
        M0 = M0@Mi
    R = np.abs(M0[1, 0]/M0[0, 0])
    return R*R

