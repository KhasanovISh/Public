# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:29:04 2023

@author: Leon
"""

def EMA(n_medium, n_inclusion, fraction_of_inclusion):
	n_m = n_medium
	n_i = n_inclusion
	d = fraction_of_inclusion
	n_eff = n_m * (2 * d * (n_i - n_m) + n_i + 2 * n_m) / (2 * n_m + n_i - d *(n_i - n_m))
	return n_eff