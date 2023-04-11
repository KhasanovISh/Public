# -*- coding: utf-8 -*-
"""
Created on Sat Apr 1

Physical model calculated here

@author: Leon
"""

# local imports
import sys
sys.path.append('E:\Python\SPPPy (1)')
from SPPPy import ExperimentSPR, Layer, MaterialDispersion
from optics import *
from helper import *
from settings import *

# general imports
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# from Helper2 import horizontal_slice_image, running_mean, hdr2

# imports packages for optimization problem
from scipy.optimize import minimize_scalar
# from functools import partial
# import seaborn as sns
# import pybobyqa

