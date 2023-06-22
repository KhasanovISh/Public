# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:24:08 2022.

Additional type classes and procedures for SPPPy

@author: THzLab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy.lib import scimath as SM
from scipy.optimize import minimize, minimize_scalar
from sympy import *
from scipy import interpolate
from types import FunctionType

class Layer:
    """Container fpr layer parametrs"""

    def __init__(self, n, thickness, name=None):
        """New layer.

        Parameters
        ----------
        n : float, complex or type
            tepe for layer.
        thickness : float
            thickness of a layer.
        name : strinf, optional
            name of a layer. The default is None.
        """
        self.n = n
        self.thickness = thickness
        self.name = name
        
    def __repr__(self):
        """Magic representation."""
        return '\n - Layer: ' + str(self.n) + ', with d ' + str(self.thickness) + "\n"


class MaterialDispersion:
    """Metall layer with complex refractive index."""

    def __init__(self, metall, base_file=None):
        """Create new metall with complex refractive index.

        Parameters
        ----------
        metall : string
            Name of a metall in base.
        base_file : string, optional
            Path to not default file. The default is None.
        """
        if base_file is None:
            base_file = "MetPermittivities.csv"
        self.name = metall

        # Dig for a data
        Refraction_data = pd.read_csv(base_file, sep=',', index_col=0)
        Refraction_data = Refraction_data[Refraction_data["Element"] == metall][[
            "Wavelength", "n", "k"]].to_numpy()

        # Get scope of definition
        self.min_lam = Refraction_data[0][0]
        self.max_lam = Refraction_data[-1][0]

        self.n_func = interpolate.interp1d(
            Refraction_data[:, 0], Refraction_data[:, 1])
        self.k_func = interpolate.interp1d(
            Refraction_data[:, 0], Refraction_data[:, 2])

    def __repr__(self):
        """Magic representation."""
        return "dispersion, \"" + self.name + "\""

    def CRI(self, wavelength):
        """Take complex refractive index.

        Parameters
        ----------
        wavelength : float
            wavelength.

        Returns
        -------
        complex
            Complex refractive index on given wavelength.
        """
        if wavelength*1e6 >= self.min_lam and wavelength*1e6 <= self.max_lam:
            return (self.n_func(wavelength*1e6) + self.k_func(wavelength*1e6)*1j)
        else:
            print("Wavelength is out of bounds!")
            print(f"CRI for {self.name} defined in: [{self.min_lam},{self.max_lam}]µm, and given: {wavelength*1e6}µm")
            if wavelength*1e6 <= self.min_lam:
                return (self.n_func(self.min_lam) + self.k_func(self.min_lam))
            if wavelength*1e6 >= self.max_lam:
                return (self.n_func(self.max_lam) + self.k_func(self.max_lam))                       

    def Show_CRI(self, lambda_range=None):
        """Plot complex refractive index.

        Parameters
        ----------
        lambda_range : array
            Range. The default is None - all data.
        """
        fig, ax = plt.subplots()
        ax.grid()
        

        if lambda_range is not None:
            if lambda_range[0]*1e6 < self.min_lam:
                print(f"Minimal bound ({lambda_range[0]*1e6}) is out of range ({self.min_lam} µm)")
                return
            if lambda_range[-1]*1e6 > self.max_lam:
                print(f"Minimal bound ({lambda_range[-1]*1e6}) is out of range ({self.mx_lam} µm)")
                return
            n_range = np.linspace(lambda_range[0]*1e6, lambda_range[-1]*1e6, len(lambda_range))
        else: n_range = np.linspace(self.min_lam, self.max_lam, 500)

        nnn = [self.n_func(j) for j in n_range]
        kkk = [self.k_func(j) for j in n_range]

        ax.plot(n_range, nnn, label='n')
        ax.plot(n_range, kkk, label='k')
        plt.title(f'Complex refractive index of {self.name}')
        plt.legend(loc='best')
        plt.ylabel('Value')
        plt.xlabel('Wavelength, µm')
        plt.show()

    def Metals_List(self, base_file=None):
        """Show all metals with definition range in wavelength.

        Parameters
        ----------
        base_file : string, optional
            Path to not default file. The default is None.
        """
        if base_file is None:
            base_file = "MetPermittivities.csv"

        Refraction_data = pd.read_csv(base_file, sep=',', index_col=0)
        
        agg_func_selection = {'Wavelength': ['min', 'max']}
        print(Refraction_data.sort_values(["Element", "Wavelength"], ascending=[True,
                         True]).groupby(['Element']).agg(agg_func_selection))


class Anisotropic:
    """Anisotropic dielectric layer."""

    def __init__(self, n0, n1, main_angle):
        """Anisotropic layer.

        Parameters
        ----------
        n0 : float
            ordinary reflection coeficient.
        n1 : float
            extraordinary reflection coeficient.
        main_angle : float
            Principle axis angle in degree
        """
        self.n0 = n0
        self.n1 = n1
        self.main_angle = np.pi * main_angle / 180

        # Equivalent rafractive indices
        self.ny_2 = (n0 * np.cos(self.main_angle))**2 + (n1 * np.sin(self.main_angle))**2
        self.nz_2 = (n0 * np.sin(self.main_angle))**2 + (n1 * np.cos(self.main_angle))**2
        self.nyz = (n0**2 - n1**2) * np.sin(self.main_angle) * np.cos(self.main_angle)

        # wave vector blocks
        self.kz_dot = lambda beta, k0: SM.sqrt(k0**2 * self.ny_2 -
                          beta**2 * self.ny_2 / self.nz_2)
        self.K = SM.sqrt(1 - self.nyz**2 / (self.ny_2 * self.nz_2))
        self.deltaK = lambda beta: (beta * self.nyz) / self.nz_2

    def kz_plus(self, beta, k0):
        """Kz+."""
        return self.kz_dot(beta, k0) * self.K + self.deltaK(beta)

    def kz_minus(self, beta, k0):
        """Kz-."""
        return self.kz_dot(beta, k0) * self.K - self.deltaK(beta)

    def r_in(self, n_prev, beta, k0):
        """r01."""
        a = n_prev**2 / SM.sqrt(n_prev**2 - (beta/k0)**2)
        b = self.p_div_q(beta, k0)
        return - (a - b) / (a + b)

    def r_out(self, n_next, beta, k0):
        """r12."""
        a = self.p_div_q(beta, k0)
        b = n_next**2 / SM.sqrt(n_next**2 - (beta/k0)**2)
        return - (a - b) / (a + b)

    def p_div_q(self, beta, k0):
        """p/q for rij."""
        return SM.sqrt(self.ny_2 * self.nz_2) * self.K / SM.sqrt(self.nz_2 - (beta/k0)**2)

    def __repr__(self):
        """Magic representation."""
        return "anisotropic, n=(" + str(self.n0) + ", " + str(self.n1) + "), angle=" + str(180*self.main_angle/np.pi)


# ------------------------------------------------------------------------
# ----------------------- Other functions --------------------------------
# ------------------------------------------------------------------------


def n_profile(func, name=None, dpi=None):
    """Parameters.

    func : function
        form of gradient layer in [0,1].
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    n_range = np.linspace(0, 1, 50)
    nnn = func(n_range)
    ax.plot(n_range, nnn)
    if name is None:
        plt.title('Gradient layer profile Shape')
    else:
        plt.title(name)
    plt.ylabel('n')
    plt.xlabel('d,%')
    plt.show()


def plot_graph(x, y, name='Reflection', tir_ang=None, label=None, dpi=None):
    """Parameters.

    x : array(float)
        x cordinates.
    y : array(float)
        x cordinates.
    name : string
        plot name..
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if label is not None:
        ax.plot(x, y, label=label)
        plt.legend(loc='best')
    else:
        ax.plot(x, y)
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.title(name)
    plt.ylim([0, 1.05])
    plt.ylabel('R')
    plt.xlabel('ϴ')
    plt.show()


def plot_2d(x, y, name='Plot', label=None, dpi=None):
    """Parameters.

    x : array(float)
        x cordinates.
    y : array(float)
        x cordinates.
    name : string
        plot name.
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if label is not None:
        ax.plot(x, y,label=label)
        plt.legend(loc='best')
    else:
        ax.plot(x, y)
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.title(name)

    plt.ylabel('R')
    plt.xlabel('ϴ°')
    plt.show()


def multiplot_graph(plots, name='Plot', tir_ang=None, dpi=None):
    """Parameters.

    plots : array(x, y, name)
        like in "Plot_Graph".
    name : string
        plot name.
    tir_ang : int, optional
        Total internal reflection angle. The default is None.
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if len(plots[0]) == 3:
        for i in plots:
            ax.plot(i[0], i[1], label=i[2])
    elif len(plots[0]) == 4:
        for i in plots:
            ax.plot(i[0], i[1], label=i[2], linestyle=i[3])
    else:
        print('Not valid array dimension')

    plt.legend(loc='best')
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.ylabel('R')
    plt.xlabel('ϴ°')
    plt.title(name)
    plt.show()


def profile_analyzer(Refl, theta_range):
    """Find SPP angle and dispersion halfwidth.

    Parameters.
    Refl : array[float]
        reflection profile.
    theta_range : range(start, end, seps)
        Range of function definition.

    Returns.
    -------
    xSPPdeg : float
        SPP angle in grad.
    halfwidth : float
        halfwidth.
    """
    div_val = (theta_range.max() - theta_range.min())/len(Refl)

    # minimum point - SPP
    yMin = min(Refl)
    # print('y_min ',yMin)
    xMin,  = np.where(Refl == yMin)[0]
    # print('x_min ',xMin)
    xSPPdeg = theta_range.min() + div_val * xMin

    # first maximum before the SPP
    Left_Part = Refl[0:xMin]
    if len(Left_Part) > 0:
        yMax = max(Left_Part)
    else:
        yMax = 1
    left_border = 0
    right_border = Refl
    half_height = (yMax-yMin)/2
    point = xMin
    while (Refl[point] < yMin + half_height):
        point -= 1
    left_border = point
    # print('left hw ', left_border)
    point = xMin
    while (Refl[point] < yMin + half_height):
        point += 1
    right_border = point
    # print('rigth hw ', right_border)

    halfwidth = div_val * (right_border - left_border)

    # print('xSPPdeg = ', xSPPdeg, 'halfwidth ', halfwidth)
    return xSPPdeg,  halfwidth

