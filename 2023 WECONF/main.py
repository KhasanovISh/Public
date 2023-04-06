# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:55:05 2023

@author: Leon

Source codes for drawing figures in the article for the conference WECONF-2023
[WECONF - Международная научная конференция](https://weconf-guap.ru/)

"""

from helper import *
from Figure1 import plot_and_save_figure_dry_measurement
from Figure2 import plot_and_save_figure_wet_measurement

import os

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__),"data.zip")
    dest_path = os.path.join(os.path.dirname(__file__),"data")
    unzip_file(data_path, dest_path)
    # plot_and_save_figure_dry_measurement() # Figure 2a
    plot_and_save_figure_wet_measurement() # Figure 2b