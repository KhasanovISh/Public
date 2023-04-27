# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:33:55 2022

@author: Leon
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import imageio

from tqdm import tqdm

import dataclasses
from wavelen2rgb import wavelen2rgb
from measurment import MeasurmentController, Measurment, MeasurmentSPR, Spectrum, smooth, plot_reflection_in_color
import traceback
from PIL import Image
from scipy.signal import argrelextrema

def filter(r):
    filtred_r = []
    # plt.imshow(r)
    # plt.show()
    def filter_line(number):
        line = smooth(r[number])
        line[line>0.5] = 0.5
        peaks = np.diff(np.sign(np.diff(line)))
        local_max_pos = np.where(peaks == -2)[0]
        line = r[number]
        if len(local_max_pos) == 1:
            # plt.title(number)
            # plt.plot(line)
            # plt.scatter(local_max_pos, line[local_max_pos])    
            for p in range(int(local_max_pos)-10, len(line)):
                if line[p] < 0.9:
                    line[p] = 1
            # plt.plot(line)                    
            # plt.show()
        filtred_r.append(np.array(line))
    for n in range(len(r)):
        filter_line(n)
    return np.array(filtred_r)


import seaborn as sns

if __name__ == '__main__':
    try:
        asbolute_path = r"E:\Experiments\2022-08-23 AO SPR3"
        folders = ['2 prism glass 192mm', 
                   '3 prism Ag 55nm 0nm 192mm', 
                   '1 prism 5 nm 192mm', 
                   '4 prism 13nm 192mm']
        
        folders = ['12 prism glass 273mm p-pol',
        '11 prism 0nm 273mm p-pol',
        '16 prism 5nm 273mm p-pol',
        '13 prism 13nm 273mm p-pol',
        '15 prism 17nm 273mm p-pol',
        '14 prism 30nm 273mm p-pol']

        experiment = MeasurmentController()
        experiment.scan(asbolute_path, folders)
        
        IMAGE_HEIGHT = 360
        IMAGE_WIDTH = 480
        
        angles = np.linspace(44,48,IMAGE_WIDTH)
        
        # for i in [1,2,3,4,5]:
        #     measurments = []
        #     for n in tqdm(range(43,0,-1)):    
        #         m0 = experiment.get_measurment(n, folders[0])
        #         m2 = experiment.get_measurment(n, folders[i])            
        #         m = MeasurmentSPR(m0, m2)
        #         if m:
        #             # plt.plot(angles, m.get_reflection(angles), label=str(i))
        #             measurments.append(m)
        #     # plt.title(n)
        #     # plt.legend()
        #     # plt.show()
        #     spectrum =  Spectrum(measurments)
        #     wls = spectrum.get_wavelengths() 
        #     wls_min = 515 # np.min(wls)
        #     wls_max = 750 # np.max(wls)
        #     wls = np.linspace(wls_min, wls_max, IMAGE_HEIGHT)
        #     s = spectrum.get_reflection(angles, wls)
        #     plt.imshow(s)
        #     plt.show()
        #     np.save("reflection " + folders[i] + ".npy", s)
        #     np.save("wavelengths " + folders[i] + ".npy", wls)
        #     np.save("angles " + folders[i] + ".npy", angles)
        #     spectrum.plot_color_spectrum(angles, wls)
        #     # spectrum.save_color_spectrum(angles, wls, folders[i] + ".npy")
        
        for i in [1,2,3,4,5]:
            result_folder  = "results\\"
            wls = np.load(result_folder + "wavelengths " + folders[i] + ".npy")
            wls = np.flip(wls)
            angles = np.linspace(43.1,47.1,IMAGE_WIDTH)
            # angles = np.flip(angles)
            # angles = np.load("angles " + folders[i] + ".npy")
            
            
            # for i in [1,2,4,5]:
            #     r = np.load("reflection " + folders[i] + ".npy")
            #     im = Image.fromarray(r)
            #     img = Image.fromarray(np.uint8(r * 255) , 'L')
            #     img.show()
            #     img.save("reflection " + folders[i] + ".bmp")
            
            img = Image.open(result_folder + "reflection " + folders[i] + " 2.bmp")
            r = np.array(img)/255
            # img.show()
            # r = filter(r)
            # spectrum = np.load(folders[3] + ".npy")
            
            # plt.imshow(r)
            # plt.show()
            
            sns.set_style("ticks")
            
            plot_reflection_in_color(r, angles, wls)
            
            plt.savefig("reflection color " + folders[i] + ".jpg", 
                        dpi=400, 
                        bbox_inches='tight')
            plt.show()
        

    except Exception:
        # traceback.print_exc(limit=1)
        raise
        # exit(1)
    