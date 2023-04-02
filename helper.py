# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 23:38:33 2023

@author: Leon
"""

import os
from joblib import Memory
import zipfile
from tqdm import tqdm

def memoize_to_file():
    mem = Memory('cached_data')
    def memoize(func):
        data = mem.cache(func)
        def wrapper(*args, **kwargs):
            result = data(*args, **kwargs)
            return result
        return wrapper
    return memoize

import matplotlib.pyplot as plt
from settings import IMAGE_DPI

def savefig(name):
    plt.savefig(name + ".jpg", 
                dpi=IMAGE_DPI, 
                bbox_inches='tight')
    

def unzip_file(file_path: str, extract_path: str) -> None:
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # Get a list of all the files in the zip
        file_list = zip_ref.namelist()
        unzipped = True
        for file in file_list:
            path = os.path.join(extract_path,file)
            # print(os.path.isfile(path))
            if not os.path.exists(path):
                unzipped = False
        if not unzipped:
            # Extract all the files while displaying a progress message
            for index, file in enumerate(tqdm(file_list, unit=' file', unit_scale=True)):
                # print(f"Extracting {file} ({index+1}/{len(file_list)})")
                zip_ref.extract(file, extract_path)
            print("Extraction completed.")