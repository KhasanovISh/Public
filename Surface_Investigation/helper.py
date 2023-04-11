# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 23:38:33 2023

@author: Leon
"""

import os
from joblib import Memory
import zipfile
from tqdm import tqdm
import requests
import os
import errno
import subprocess
import numpy as np

# __all__ = ['PublicClass']

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
            

def download_figshare_zip(url, output_file):
    if os.path.isfile(output_file):
        print(f"File {output_file} already exists!")
        return
    else:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(output_file, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
                
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR: Download incomplete.")
            
def get_files_with_extension(folder_path, extension):
    file_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_list.append(os.path.join(folder_path, file_name))
    return file_list

def resave_channel_3to1(path):
    files = get_files_with_extension(path, ".npy")
    for file in tqdm(files):
        np.save(file, np.load(file)[:,:,0])