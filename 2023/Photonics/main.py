# -*- coding: utf-8 -*-
"""
Created on 2023-05-05 14:00:04
@author: I. Khasanov
"""

from model import YeatmanDynamicModel

def main():
    # Plot the model reproduced from Yeatman (1996) doi:10.1016/0956-5663(96)83298-2
    # altered with Gaussian light source 
    # with dynamic change of the incident light wave vector with a slider  
    YeatmanDynamicModel() 

if __name__ == '__main__':
    main()