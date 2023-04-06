# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:31:08 2023

@author: Leon
"""

# local imports
import sys
sys.path.append('E:\Python\SPPPy (1)')
from SPPPy import ExperimentSPR, Layer, MaterialDispersion, plot_graph

# general imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Helper2 import horizontal_slice_image, running_mean, hdr2

# imports packages for optimization problem
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import seaborn as sns
import pybobyqa

# UNITS
nm = 1e-9
um = 1e-6

def savefig(name):
    plt.savefig(name + ".jpg", 
                dpi=400, 
                bbox_inches='tight')

def get_glass_RI(wavelength_nm, filename):
    glass_dispersion = np.genfromtxt(filename)
    glass_wl, glass_n = list(zip(*glass_dispersion))
    return np.interp(wavelength_nm/1000, glass_wl, glass_n)

def setup_AuSPR_theoretical(wavelength_nm, d_nm = 0, n = 1):
    glass_refractive_index = get_glass_RI(wavelength_nm, 
                                          'Glass LK-7 dispersion.dat')
    # n_Ag = 0.15 + 4.819j
    d_Au = 50
    AgSPR = ExperimentSPR(polarisation='p')
    AgSPR.wavelength = wavelength_nm * nm
    AgSPR.add(Layer(glass_refractive_index, 1))
    # AgSPR.add(Layer(n_Ag, d_Ag*1e-9))
    AgSPR.add(Layer(MaterialDispersion("Au"), d_Au * nm))
    AgSPR.add(Layer(n, d_nm * nm))
    AgSPR.add(Layer(1, 1))
    return AgSPR

def get_ethanol_RI(wavelength_nm):
    x = wavelength_nm/1000
    n=1.35265+0.00306*x**-2+0.00002*x**-4
    return n

def get_wavelength(folder, number):
    spectrum = np.genfromtxt(folder.joinpath(f"{number}.DAT"), skip_header = 7)
    return spectrum[np.where(spectrum[:,1] == np.max(spectrum[:,1]))[0][0]][0]

def get_spectral_angle_curve(folder):
    folder = Path(folder)
    numbers = np.arange(68,82) #93
    exposures = np.array([1000, 3205, 6024, 10000])
    exposure = exposures[3]
    wavelengths=[]
    positions = []
    for number in numbers:
        wavelengths.append(get_wavelength(folder, number))
        path = folder.joinpath(f"{number}_experiment_exposure_{exposure}.npy")
        image = np.load(path)[:,:,0]
        start = 200
        curve = image[700][start:1100]
        positions.append(np.mean(np.where(curve == np.min(curve))[0]+start))
    return numbers, wavelengths, positions

def get_SPR_minima(wavelength_nm, d_nm = 0, refractive_index = 1):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= refractive_index)
    # angles = np.linspace(AuSPR.TIR(),60,1000)
    bnds = (AuSPR.TIR(),60)
    get_R = lambda x: AuSPR.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

    

def get_linear_scale_params(x,y):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def ZrO2(wl_nm):
    x = wl_nm/1000
    return (1+1.347091/(1-(0.062543/x)**2)+2.117788/(1-(0.166739/x)**2)+9.452943/(1-(24.320570/x)**2))**.5

def TiO2(wl_nm):
    x = wl_nm/1000
    return (5.913+0.2441/(x**2-0.0803))**.5

def SiO2(wl_nm):
    x = wl_nm/1000
    return (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5

    

def EMA(n_medium, n_inclusion, fraction_of_inclusion):
	n_m = n_medium
	n_i = n_inclusion
	d = fraction_of_inclusion
	n_eff = n_m * (2 * d * (n_i - n_m) + n_i + 2 * n_m) / (2 * n_m + n_i - d *(n_i - n_m))
	return n_eff


def gradient(x, wl_nm, func):
    return EMA(SiO2(wl_nm), EMA(ZrO2(wl_nm),TiO2(wl_nm),0.4), func(x))

def linear(x, wl_nm):
    return EMA(SiO2(wl_nm), EMA(ZrO2(wl_nm),TiO2(wl_nm),0.4), x*x)

def inv_linear(x, wl_nm):
    return EMA(EMA(ZrO2(wl_nm),TiO2(wl_nm),0.4), SiO2(wl_nm), x*x)

def get_TIR():
    AuSPR = setup_AuSPR_theoretical(600, d_nm= 0, n= 1)
    return AuSPR.TIR()

def get_SPR_minima_gradient(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    bnds = (AuSPR.TIR(),60)
    AuSPR.layers[2] = Layer(lambda x: gradient(x,wavelength_nm, gradient_func), d_nm*nm)
    # angles = np.linspace(AuSPR.TIR(),60,1000)
    get_R = lambda x: AuSPR.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

def get_SPR_minima_gradient2(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    bnds = (AuSPR.TIR(),60)
    AuSPR.layers[2] = Layer(lambda x: gradient_func(x), d_nm*nm)
    # angles = np.linspace(AuSPR.TIR(),60,1000)
    get_R = lambda x: AuSPR.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

def get_SPR_minima_gradient_new(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    bnds = (AuSPR.TIR(),90)
    AuSPR.layers[2] = Layer(lambda x: gradient_func(x), d_nm*nm)
    AuSPR.layers[3] = Layer(get_ethanol_RI(wavelength_nm), 100)
    # angles = np.linspace(AuSPR.TIR(),60,1000)
    get_R = lambda x: AuSPR.R(angles=[x])[0]
    res = minimize_scalar(get_R,
                    method='bounded', bounds=bnds)
    return res.x

def get_SPR_curve_gradient(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    AuSPR.layers[2] = Layer(lambda x: gradient(x,wavelength_nm, gradient_func), d_nm*nm)
    
    degree_range = np.linspace(42,45,100)
    R = AuSPR.R(angles=degree_range)
    return degree_range, R

def get_SPR_curve_gradient2(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    AuSPR.layers[2] = Layer(lambda x: gradient_func(x), d_nm*nm)
    
    degree_range = np.linspace(42,45,100)
    R = AuSPR.R(angles=degree_range)
    return degree_range, R

def get_SPR_curve_gradient_new(wavelength_nm, d_nm = 0, gradient_func = None):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= 1)
    AuSPR.layers[2] = Layer(lambda x: gradient_func(x), d_nm*nm)
    
    degree_range = np.linspace(42,45,100)
    R = AuSPR.R(angles=degree_range)
    return degree_range, R

def get_SPR_curve(wavelength_nm, d_nm = 0, n = 1):
    AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= d_nm, n= n)    
    degree_range = np.linspace(42,45,100)
    R = AuSPR.R(angles=degree_range)
    return degree_range, R

def experimental_gradient_SPR():
    asbolute_path = r"E:\Experiments\2022-12-17"
    folders = ['Au 2 0nm', 
               'Au 2 grad']
    
    
    folder = Path(asbolute_path).joinpath(folders[0])
    numbers, wavelengths, positions = get_spectral_angle_curve(folder)
    
    
    mes_grad_numbers = np.array([77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88])
    mes_grad_positions = np.array([444, 459, 476, 507, 530, 566, 599, 631, 671, 747, 784, 810])    
    
    thetas2 = []
    for wl in wavelengths:
        thetas2.append(get_SPR_minima(wl))
   
    m,c = get_linear_scale_params(positions, thetas2)
    # plt.plot(positions, thetas, 'o', label='Original data', markersize=10)
    # plt.plot(positions, m*np.array(positions) + c, 'r', label='Fitted line')
    # plt.legend()
    
    mes_grad_wavelengths = []
    for num in mes_grad_numbers:
        wl = get_wavelength(folder, num)
        mes_grad_wavelengths.append(wl)
    
    plt.plot(wavelengths, thetas2,  label = "1: Au")
    plt.plot(mes_grad_wavelengths, m*mes_grad_positions + c, "b.", label = "2: Au+град.")
    plt.xlabel("Длина волны λ, нм")
    plt.ylabel("Резонансный угол θ, $^{o}$") 
    plt.legend()
    savefig("Измерения")
    plt.show()
    return mes_grad_wavelengths, m*mes_grad_positions + c

def main():

    mes_grad_wavelengths, mes_grad_thetas  = experimental_gradient_SPR()

    # for n in [1.65,1.66]:
    #     thetas = []
    #     for wl in mes_grad_wavelengths:
    #         thetas.append(get_SPR_minima(wl, 220, n))
    #     plt.plot(mes_grad_wavelengths, thetas, label = f"{n}")
        
    TIR = get_TIR()
    def plot_gradient(func, label):
        thetas = []
        for wl in mes_grad_wavelengths:
            thetas.append(get_SPR_minima_gradient(wl, 220, func))
        thetas = np.array(thetas)
        thetas[thetas <= TIR+0.1] = None
        plt.plot(mes_grad_wavelengths, thetas, label = label)
    
    grad_funcs = [lambda x: x, lambda x: 1-x, lambda x: x*x, lambda x: 1-x*x,
                  lambda x: x*x*x, lambda x: 1-x*x*x,lambda x: x*x- x + 1, lambda x: -x*x + x + 1,
                  lambda x:  (-0.74565806)*np.sin(0.5*np.pi*x)+2.11390685,
                  lambda x: -0.16717102 *x*x  +1.70482002,
                  lambda x: -0.0722967 *x*x*x + 1.67035575,
                  lambda x: -0.49352205* x + 1.89009383,
                  lambda x: x*0+1.65476709]
    grad_funcs_names = ["лин.","обр. лин.","квадр.","обр. квадр.",
                        "куб.","обр. куб.", "квадр.","обр. квадр.",
                        "sin", "$x^2$", "$x^3$", "x", "1.65"] 
    
    # for i in range(len(grad_funcs)):
    #     plot_gradient(grad_funcs[i], f"{i+3}: "+grad_funcs_names[i]) 
    
    
    
    
    
    # plt.ylabel("Коэф-т отражения $R_p$")
    # plt.xlabel("Резонансный угол θ, $^{o}$")
    
    # degree_range = np.linspace(42.5, 43, 500)
    degree_range = np.linspace(51, 55, 500)
    
    def plot_SPR(num, wl):
        AuSPR = setup_AuSPR_theoretical(wl, d_nm= 220*nm, n = 1)
        AuSPR.layers[2] = Layer(lambda x: grad_funcs[num](x), 220*nm)
        data = AuSPR.R(angles=degree_range)
        plt.plot(degree_range, data, label=grad_funcs_names[num])
    
    # plot_SPR(12, 650)
    # plot_SPR(11, 500)
    # plot_SPR(9, 500)
    # plot_SPR(10, 500)
    # plot_SPR(8, 500)
    # plt.legend()
    # savefig(f"Сравнение ППР 500")
    # plt.show()
    
    # "o", label = "0",
    
    grad_funcs = [lambda x: x*0+1.65476709,
                  lambda x: -0.49352205* x + 1.89009383,
                  lambda x: -0.16717102 *x*x  +1.70482002,
                  lambda x: -0.0722967 *x*x*x + 1.67035575
                  ]
    
    Ns = [0,1,2,3]
    styles = ["r--","g--","b--","k--"]
    labels = ["1","2","3","4"]
    
    # plt.plot(mes_grad_wavelengths, mes_grad_thetas, "ko", label = "0")

    # for i, N in enumerate(Ns):
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
    # savefig("первое измерение")
    # plt.show()
    
    
    for i, N in enumerate(Ns):
        thetas = []
        for wl in mes_grad_wavelengths:
            thetas.append(get_SPR_minima_gradient_new(wl, 220, grad_funcs[N]))
        thetas = np.array(thetas)
        thetas[thetas <= TIR+0.1] = None
        plt.plot(mes_grad_wavelengths, thetas, styles[i], label = f"{i+1}")
    # plt.errorbar(mes_grad_wavelengths, mes_grad_thetas, yerr=0.15, fmt = "o", label = "0", capsize=2)
    plt.xlabel("Длина волны λ, нм")
    plt.ylabel("Резонансный угол θ, $^{o}$")
    plt.legend()
    savefig("второе измерение")
    plt.show()
    
    
    ############################################
    
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(mes_grad_wavelengths, mes_grad_thetas, ".", label = "Au+град.")
    
    # N = 8
    # thetas = []
    # for wl in mes_grad_wavelengths:
    #     thetas.append(get_SPR_minima_gradient2(wl, 220, grad_funcs[N]))
    # thetas = np.array(thetas)
    # thetas[thetas <= TIR+0.1] = None
    # ax1.plot(mes_grad_wavelengths, thetas, "k-", label = grad_funcs_names[N])
    # ax1.set(xlabel="Длина волны λ, нм", ylabel="Резонансный угол θ, $^{o}$")
    # ax1.set_title('Резонансные кривые')
    # ax1.set_box_aspect(0.75)
    # ax1.legend()
    
    # xrange = np.linspace(0,1,100)
    # ax2.plot([x for x in xrange*220], [grad_funcs[N](x) for x in xrange], "k-")
    # ax2.set_box_aspect(0.75)
    # ax2.set_title('Оптический профиль')
    # ax2.set(xlabel="Толщина слоя z, нм", ylabel="Показатель преломления n")
    
    
    # plt.subplots_adjust(wspace=0.4)
    # savefig(f"Сравнение {N}")
    # plt.show()
    
    def plot_gradient_curve(func, label):
        for wl in [mes_grad_wavelengths[1]]:
            th, R = get_SPR_curve_gradient(wl,220,func)
            plt.plot(th,R)
        # plt.title(label)
        # plt.show()
    
    # for wl in [mes_grad_wavelengths[1]]:
    #     th, R = get_SPR_curve(wl,220, 1.67)
    #     plt.plot(th,R)
    # plt.title("конст.")
    # # plt.show()
    
    # for i in [2]: #range(len(grad_funcs))
    #     plot_gradient_curve(grad_funcs[i],grad_funcs_names[i])

        
    # AuSPR = setup_AuSPR_theoretical(650)
    # degree_range = np.linspace(40,60,100)
    # R = AuSPR.R(angles=degree_range)
    # plt.plot(degree_range, R)
    # plt.show()  

def lowhigh(x):
    if hasattr(x, '__iter__'):
        profile = []
        for y in x:
            if y < 0.5:
                profile.append(1.5) 
            else:
                profile.append(2)
        return profile
    else:
        return 1.5 if x < 0.5 else 2

def test():
    wavelength_nm = 650
    glass_refractive_index = get_glass_RI(wavelength_nm, 
                                          'Glass LK-7 dispersion.dat')
    # n_Ag = 0.15 + 4.819j
    d_Au = 50
    AgSPR = ExperimentSPR()
    AgSPR.wavelength = wavelength_nm * nm
    AgSPR.add(Layer(glass_refractive_index, 1))
    # AgSPR.add(Layer(n_Ag, d_Ag*1e-9))
    AgSPR.add(Layer(MaterialDispersion("Au"), d_Au * nm))
    AgSPR.add(Layer(1.5, 110 * nm))
    AgSPR.add(Layer(2, 110 * nm))
    AgSPR.add(Layer(1, 1))
    
    degree_range = np.linspace(40, 60, 200)
    R = AgSPR.R(angles=degree_range)
    plt.plot(degree_range, R, label="low > high")
    
    AgSPR.layers[2] = Layer(2, 110 * nm)
    AgSPR.layers[3] = Layer(1.5, 110 * nm)
    R = AgSPR.R(angles=degree_range)
    plt.plot(degree_range, R, label="high > low")
    

    
    AgSPR.layers[2] = Layer(lambda x: lowhigh(x), 220 * nm)
    AgSPR.layers[3] = Layer(1, 110 * nm)
    R = AgSPR.R(angles=degree_range)
    plt.plot(degree_range, R, label="grad low > high")
    plt.legend()
    plt.show()
    




def optimize_func(wavelengths, thetas):
    def gradient_func(p, x):
        # return p[0]*np.sin(0.5*np.pi*x)+p[1]
        # return p[0]*x*x+p[1]
        # return p[0]*x*x*x+p[1]
        # return p[0]*x+p[1]
        # return 0*x+p[1]
        return p[0]*x*x+p[1]*x + p[2]

    
    thetas = np.array(thetas)
    print(thetas)
    print(wavelengths)
    plt.show()
    AuSPR = setup_AuSPR_theoretical(650, d_nm= 220*nm, n = 1)
    TIR = AuSPR.TIR()
    
    def get_SPR_minima2(wavelength_nm, parameters = None):
        AuSPR = setup_AuSPR_theoretical(wavelength_nm, d_nm= 220*nm, n = 1)
        bnds = (AuSPR.TIR(),60)
        AuSPR.layers[2] = Layer(lambda x: gradient_func(parameters, x), 220*nm)
        # degree_range = np.linspace(40, 60, 200)
        # data = AuSPR.R(angles=degree_range)
        # plot_graph(degree_range, data, dpi=50)
        # angles = np.linspace(AuSPR.TIR(),60,1000)
        get_R = lambda x: AuSPR.R(angles=[x])[0]
        res = minimize_scalar(get_R,
                        method='bounded', bounds=bnds)
        return res.x
    
    def check_constraints_is_ok(x):
        """
            проверяем укладывается ли значения полинома, описывающего 
            оптический профиль рградиентного слоя в границы физически осмысленного
            интервала показателя преломлений от 1 до UPPER_LIMIT
        """
        UPPER_LIMIT = 3    
        # ищем корни полинома, которые пересекают нижнюю границу (равную 1)
        # эти корни должны лежать вне отрезка от 0 до 1 (координата глубины слоя)
        coeff = x.copy()
        coeff[0] = coeff[0]+1
        roots = np.roots(coeff)
        for root in roots:
            if 0 < root < 1:
                return False
            
        # значения должны быть положительны
        if np.poly1d(x)(0.5) < 0:
            return False
        
        # экстремумы полинома должны лежать внутри интервала
        derivative_coeff = np.polyder(x)
        roots = np.roots(derivative_coeff)
        for root in roots:
            if 0 < root < 1:
                if np.poly1d(derivative_coeff)(root) > UPPER_LIMIT:
                    return False

        # на границах отрезка значения должны лежать внутри интервала
        if np.poly1d(x)(0) > UPPER_LIMIT:
            return False
        if np.poly1d(x)(1) > UPPER_LIMIT:
            return False
        return True
    
    def sin_gradient(p):
        if not check_constraints_is_ok(p):
            print(f"overflow {p}")
            return 10000
        print(p)
        ths = []
        for i in range(len(wavelengths)):
            th = get_SPR_minima2(wavelengths[i], p)
            # if th <= TIR+0.1:
            ths.append(th)
            # else:
            #     ths.append(thetas[i])
        ths = np.array(ths)
        dth = ths - thetas
        dth = np.sum(dth*dth)
        # print(ths)
        print(dth)
        plt.plot(wavelengths, thetas, "--", label = "Au+град.")
        plt.plot(wavelengths, ths, "-", label = f"{p}")
        plt.legend()
        plt.show()
        return dth 
    
    x0 = np.array([0, 0, 1.65])
    lower = np.array([-3, -3, -3])
    upper = np.array([3, 3, 3])
    # x0 = np.array([1.65])
    # lower = np.array([1])
    # upper = np.array([3])

    soln = pybobyqa.solve(sin_gradient, x0, bounds=(lower,upper), 
                          seek_global_minimum=True)
    print(soln)

if __name__ == '__main__':
    sns.set_style("ticks")
    
    # optimize_func(*experimental_gradient_SPR())
    
    main()
    # test()
    
    
    # xrange = np.linspace(0,1,100)
    # plt.plot([x for x in xrange*220], [gradient(x, 650, lambda x: 0.5) for x in xrange])
    # plt.xlabel("Толщина слоя z, нм")
    # plt.xticks([0,50,100,150,200,220])
    # plt.ylabel("Показатель преломления n")
    # plt.show()

