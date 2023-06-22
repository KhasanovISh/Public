# -*- coding: utf-8 -*-
"""
Created on 2023-05-29 20:33:54
@author: I. Khasanov
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    matplotlib.rcParams.update({'font.size': 16})
    x = np.linspace(0,1,100)
    plt.plot(x*0+1.655, x*220, label="$x^0$")
    plt.plot(-0.494*x  + 1.890, x*220, label="$x^1$")
    plt.plot(-0.167*x**2 +1.705, x*220, label="$x^2$")
    plt.plot(-0.072*x*x*x + 1.671, x*220, label="$x^3$")
    plt.ylabel("z, нм")
    plt.xlabel("n")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()