# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:13:47 2021

@author: Khasanov Ildus

Построение рисунков для иллюстрации примера фантомной микроскопии 
для плазмонной линзы Люнеберга 
в статье в журнал 
"Поверхность. Рентгеновские, синхронные и нейтронные исследования"

Calculations to illustrate an example of ghost SPR microscopy 
for the Luneberg plasmonic lens in an article in the Journal of 
"Surface Investigation. X-Ray, Synchrotron and Neutron Techniques".

"""

import numpy as np
from numpy.lib import scimath as SM
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from diffractio import um, nm, mm, degrees
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY

# ----------------------------------------------------------------------------
#  Utility functions / Служебные функции
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Graphics options / графические настройки
# ----------------------------------------------------------------------------

# import seaborn as sns
# sns.set()
# sns.set_theme(style="white", palette='dark')
num_pixels = 45
REGION_SIDE = 2  # extended region
EXTENT = (-REGION_SIDE ,REGION_SIDE,-REGION_SIDE,REGION_SIDE)
xrange = np.linspace(-REGION_SIDE,REGION_SIDE,num_pixels)
yrange = np.linspace(-REGION_SIDE,REGION_SIDE,num_pixels)


# ----------------------------------------------------------------------------
# Surface plasmon resonance / Поверхностный плазмонный резонанс


METAL_THICKNESS = 65e-9
LAYER_THICKNESS = 10e-9 
DEFAULT_WAVELENGTH = 632.8*1e-9

def radian(degree):
    return np.pi * degree / 180


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

def reflection(n, theta):
    # refractive index for wavelength 632.8 nm from https://refractiveindex.info
    metal = -12.033+1.1634j # Evaporated gold R. L. Olmon, B. Slovick, T. W. Johnson, D. Shelton, S.-H. Oh, G. D. Boreman, and M. B. Raschke. Optical dielectric function of gold, Phys. Rev. B 86, 235147 (2012)  get_eps(wl*1e-3, Au_n,Au_k)
    glass = 1.5151 # BK7 SHOTT 
    return reflectivity([glass, metal, n*n, 1],[0, METAL_THICKNESS, LAYER_THICKNESS ,0], radian(theta))

def luneburg_n(x,y):
    eps = 2 - (x*x+y*y)/(RL*RL)
    if eps <= 1:
        return 1
    return np.sqrt(eps)



def show(data, x1=0,x2=-1,y1=0,y2=-1, colormap = 'gist_yarg'):
    plt.imshow(data[x1:x2,y1:y2], cmap = colormap, extent=(xrange[x1],xrange[x2],yrange[y1],yrange[y2]))


# ----------------------------------------------------------------------------
# Luneburg refrective index distribution / Модель линзы Люнеберга
RL = 1.05 # Luneburg radii / радиус линзы
Luneburg_n = []
for xi in xrange:
    a = []
    for yi in yrange:
        a.append(luneburg_n(xi, yi))
    Luneburg_n.append(a)
Luneburg_n = np.array(Luneburg_n)


# 
plt.axvline(x=1.0,linestyle ='--', color='gray')
plt.axvline(x=np.sqrt(2),linestyle ='--', color='gray')
plt.xticks([1,1.25,1.41,1.6,1.8,2])
# plt.title('Зависимость коэффицента отражения \n от показателя преломления \n для плазмонной линзы Люнеберга')
plt.xlabel('n')
plt.ylabel('R')
th = [59,60,61]
colors = ['0.5','0','0.8']
labels = ['θ = '+ str(t) + '°' for t in th]
for i in [0,1,2]:
    n = np.linspace(1,2,100)
    R = [reflection(ni,th[i]) for ni in n]
    n2 = np.linspace(0.9,1,10)
    R2 = [reflection(ni,th[i]) for ni in n2]
    to_R = interp1d(n, R, fill_value="extrapolate")
    to_n = interp1d(R, n, fill_value="extrapolate")
    Luneburg_R = to_R(Luneburg_n)    
    plt.plot(n,R, color=colors[i], label=labels[i])
    plt.plot(n2,R2,'--', color=colors[i])
    
plt.legend(loc='upper right')
plt.show()

n = np.linspace(0.9,1.41,100)
R = [reflection(ni,60) for ni in n]
to_R = interp1d(n, R, fill_value="extrapolate")
to_n = interp1d(R, n, fill_value="extrapolate")
Luneburg_R = to_R(Luneburg_n)   

# ----------------------------------------------------------------------------
# GHOST IMAGING / МЕТОД ФАНТОМНЫХ ИЗОБРАЖЕНИЙ

NUMBER_OF_MEASURMENTS = 100000
length = 2 * mm
x0 = np.linspace(-length / 2, length / 2, num_pixels)
y0 = np.linspace(-length / 2, length / 2, num_pixels)
wavelength = 632.8 * nm
u1 = Scalar_source_XY(x=x0, y=y0,
                     wavelength=wavelength)
u1.gauss_beam(
    A=1,
    r0=(0 * mm, 0 * mm),
    z0=0,
    w0=(10 * mm, 10 * mm),
    phi=0 * degrees,
    theta=0 * degrees)

patterns = []
buckets = []
ref_buckets = []

# ----------------------------------------------------------------------------
# Speckle patterns generation / генерация спекл-картин

# np.random.seed(120)
# for item in progressBar(range(NUMBER_OF_MEASURMENTS), prefix = 'Progress:', suffix = 'Complete', length = 50):
#     length = 2 * mm
#     x0 = np.linspace(-length / 2, length / 2, num_pixels)
#     y0 = np.linspace(-length / 2, length / 2, num_pixels)
#     t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
#     t1.roughness(t=(500 * um, 500 * um), s=500 * um)        
#     u2 = u1 * t1
#     u3 = u2.RS(z= 1000 * mm, new_field=True)
#     speckle_image = u3.intensity()
#     patterns.append(speckle_image.astype(np.half))
#     buckets.append(np.sum(speckle_image*Luneburg_R))

# np.save("patterns.npy", patterns)



patterns = np.load('patterns.npy')

# plt.title('speckle image')
# plt.title('Пример спекл-картины')
speckle_img = patterns[0].astype(float)
show(speckle_img,10,35,10,35)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
cax = plt.axes([0.91, 0.125, 0.05, 0.755])
cmin = np.round(np.min(speckle_img[10:35,10:35]),5)
cmax = np.round(np.max(speckle_img[10:35,10:35]),3)
cticks = [0.001]+list(np.round(np.linspace(cmin,cmax,4)[1:-1],2))+[cmax]
plt.colorbar(cax=cax, ticks=cticks)
plt.show()

# """

NUMBER_OF_MEASURMENTS = 100000
patterns = patterns[0:NUMBER_OF_MEASURMENTS]

# single-pixel bucket detector / заполнение измерений однопиксельного приёмника
for p in patterns:
    buckets.append(np.sum(p*Luneburg_R))
    ref_buckets.append(np.sum(p))

# mean values calculation / расчёт средних
mean_bucket = np.mean(buckets)
mean_ref_buckets = np.mean(ref_buckets)
mean_pattern = np.mean(patterns,axis=0)
pic_size = len(mean_pattern)

# plt.title('mean pattern')
show(mean_pattern.astype(float),10,35,10,35)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
cax = plt.axes([0.91, 0.12, 0.05, 0.77])
plt.colorbar(cax=cax)
plt.show()

# ghost image calculation / вычисление корелляционной функции второго порядка
ghost_image = np.zeros((pic_size,pic_size)) 
for i in range(NUMBER_OF_MEASURMENTS):
    ghost_image += (buckets[i]-mean_bucket)*(patterns[i] - mean_pattern)
ghost_image = (ghost_image / NUMBER_OF_MEASURMENTS) * 0.9 / mean_pattern


# plt.title('ghost image')
# plt.title('Фантомное изображение')
show(ghost_image,10,35,10,35)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
cax = plt.axes([0.91, 0.125, 0.05, 0.755])
cmin = np.min(ghost_image[10:35,10:35])
cmax = np.max(ghost_image[10:35,10:35])
cticks = [cmin]+list(np.round(np.linspace(cmin,cmax,5)[1:-1],2))+[cmax]
plt.colorbar(cax=cax, ticks=cticks)
plt.show()


# plt.title('Распределение \n коэффицента отражения \n для линзы Люнеберга')
# plt.title('Reflectance distribution \n for a Luneberg lens')
show(Luneburg_R,10,35,10,35)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
cax = plt.axes([0.91, 0.125, 0.05, 0.755])
cmin = np.min(Luneburg_R[10:35,10:35])
cmax = np.max(Luneburg_R[10:35,10:35])
cticks = [cmin]+list(np.round(np.linspace(cmin,cmax,5)[1:-1],2))+[cmax]
plt.colorbar(cax=cax, ticks=cticks)
plt.show()


# plt.title('Распределение \n показателя преломления \n для линзы Люнеберга')
# plt.title('Refractive index \n distribution for a Luneberg lens')
ax = plt.subplot()
show(Luneburg_n,10,35,10,35)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
cax = plt.axes([0.91, 0.125, 0.05, 0.755])
cmin = np.min(Luneburg_n[10:35,10:35])
cmax = np.max(Luneburg_n[10:35,10:35])
cticks = [cmin]+list(np.round(np.linspace(cmin,cmax,5)[1:-1],2))+[cmax]
plt.colorbar(cax=cax, ticks=cticks)
plt.show()

# plt.title('dielelectric refractive index \n profile by ghost SPR microscopy ')
# plt.title('Распределение показателя преломления \n в фантомной микроскопии ППР')
ghost_image_n = to_n(ghost_image)
show(ghost_image_n,10,35,10,35)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
cax = plt.axes([0.91, 0.125, 0.05, 0.755])
cmin = np.min(ghost_image_n[10:35,10:35])
cmax = np.max(ghost_image_n[10:35,10:35])
cticks = [cmin]+list(np.round(np.linspace(cmin,cmax,5)[1:-1],2))+[cmax]
plt.colorbar(cax=cax, ticks=cticks)
plt.show()


# """