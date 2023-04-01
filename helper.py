# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 23:38:33 2023

@author: Leon
"""

from joblib import Memory

def memoize_to_file():
    mem = Memory('cached_data')
    def memoize(func):
        data = mem.cache(func)
        def wrapper(*args, **kwargs):
            result = data(*args, **kwargs)
            return result
        return wrapper
    return memoize