# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:00:03 2023

@author: Leon
Source codes for drawing figure in the article for the conference WECONF-2023
[WECONF - Международная научная конференция](https://weconf-guap.ru/)

"""

from settings import *
from model import *
from helper import *

import matplotlib.pyplot as plt

@memoize_to_file()
def calculate_data():
    return experimental_gradient_SPR()

def plot_figure_dry_measurement():

    mes_grad_wavelengths, mes_grad_thetas = calculate_data()
    
    plt.plot(mes_grad_wavelengths, mes_grad_thetas, "ko", label = "0")
    plt.show()

    # for i, N in enumerate(range(3)):
    #     thetas = []
    #     for wl in mes_grad_wavelengths:
    #         thetas.append(get_SPR_minima_gradient2(wl, 220, grad_funcs[N]))
    #     thetas = np.array(thetas)
    #     thetas[thetas <= TIR+0.1] = None
    #     plt.plot(mes_grad_wavelengths, thetas, styles[i], label = f"{i+1}")
    # plt.errorbar(mes_grad_wavelengths, mes_grad_thetas, yerr=0.15, color="k", fmt = "o", capsize=2)
    # plt.plot(mes_grad_wavelengths, mes_grad_thetas, "ko")
    # plt.xlabel("Длина волны λ, нм")
    # plt.ylabel("Резонансный угол θ, $^{o}$")
    # plt.legend()
    # # savefig("первое измерение")
    # plt.show()