# -*- coding: utf-8 -*-
"""
Created on Sat Apr  11 14:44 2023

@author: Leon

Source codes for drawing figures in the article in the journal Surface Investigation: X-Ray, Synchrotron and Neutron Techniques
https://www.springer.com/journal/11700

"""

from helper import *
from Figure1 import plot_1()
import os

if __name__ == '__main__':
    #data_path = os.path.join(os.path.dirname(__file__),"data.zip")
    #dest_path = os.path.join(os.path.dirname(__file__),"data")
    #unzip_file(data_path, dest_path)
    plot_1() # Figure 1a