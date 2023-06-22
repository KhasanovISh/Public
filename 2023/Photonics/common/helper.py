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
import matplotlib.pyplot as plt

# __all__ = ['PublicClass']

def memoize_to_file():
    """
    Returns a decorator that caches the output of a function to a file.

    The decorator uses a `Memory` object from the `joblib` library to cache the function's output to a file named `cached_data`. The cached data can be reused on subsequent calls to the function with the same inputs, which can improve performance.

    Usage:
        @memoize_to_file()
        def my_function(*args, **kwargs):
            ...

    Args:
        None

    Returns:
        A decorator function that can be used to cache the output of a function to a file.
    """
    mem = Memory('cached_data')
    def memoize(func):
        data = mem.cache(func)
        def wrapper(*args, **kwargs):
            result = data(*args, **kwargs)
            return result
        return wrapper
    return memoize

from settings import IMAGE_DPI

def savefig(name: str):
    """
    Save the current figure to a file with the specified name and DPI.

    This function saves the current matplotlib figure to a JPEG file with the specified name and DPI (dots per inch).
    The `bbox_inches='tight'` option is used to ensure that the bounding box of the figure is tightly cropped around its contents.

    Args:
        name (str): The name of the file to save the figure to, without the extension.

    Returns:
        None
    """
    plt.savefig(name + ".jpg", 
                dpi=IMAGE_DPI, 
                bbox_inches='tight')
    
def unzip_file(file_path: str, extract_path: str):
    """
    Extracts a zip file to the specified extract path.

    This function extracts a zip file to the specified extract path, displaying a progress bar using the `tqdm` library.

    Args:
        file_path (str): The path to the zip file to extract.
        extract_path (str): The path to extract the zip file to.

    Returns:
        None
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # Get a list of all the files in the zip
        file_list = zip_ref.namelist()
        unzipped = all(os.path.exists(os.path.join(extract_path, file)) for file in file_list)

        if not unzipped:
            # Extract all the files while displaying a progress message
            with tqdm(file_list, desc="Extracting", unit='file', unit_scale=True) as progress_bar:
                for file in progress_bar:
                    zip_ref.extract(file, extract_path)

            print("Extraction completed.")
        else:
            print("Files already exist. Skipping extraction.")
            
def download_figshare_zip(url: str, output_file: str):
    """
    Downloads a zip file from a specified URL and saves it to a file.

    This function downloads a zip file from a specified URL and saves it to a file with the specified name. The download
    progress is displayed using the `tqdm` library.

    Args:
        url (str): The URL of the zip file to download.
        output_file (str): The name of the file to save the downloaded zip file to.

    Returns:
        None
    """
    if os.path.isfile(output_file):
        print(f"File {output_file} already exists!")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0)) # in bytes
    block_size = 1024 # 1 Kibibyte
	
    with open(output_file, 'wb') as f, tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
	
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete.")
            
def get_files_with_extension(folder_path: str, extension: str):
    """
    Returns a list of all file paths in the specified directory with the specified extension.

    Args:
    - folder_path (str): A string specifying the path of the directory to search in.
    - extension (str): A string specifying the file extension to search for (e.g. '.txt').

    Returns:
    - A list of strings representing the absolute file paths of all files in the specified directory
    that have the specified file extension.

    Example:
    >>> get_files_with_extension('/path/to/folder', '.txt')
    ['/path/to/folder/file1.txt', '/path/to/folder/file2.txt']
    """
    return [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(extension)]

def resave_channel_3to1(path: str):
    """
    Resaves all '.npy' data files of three-color images in the specified directory, saving only the first channel.

    Args:
    - path (str): A string specifying the path of the directory containing '.npy' files to resave.

    Returns:
    - None.
    """
    # Get a list of all the '.npy' files in the specified directory
    files = get_files_with_extension(path, ".npy")

    # Resave each '.npy' file, saving only the first channel
    for file in tqdm(files):
        np.save(file, np.load(file)[:,:,0])

def load_expeimental_column_data(path):
    """Load experimental data from a file."""
    try:
        data = np.genfromtxt(path,
                             encoding = 'utf-8', 
                             delimiter='\t',
                             dtype=float)
        return data[:,0],data[:,1]
    except OSError:
        raise ValueError(f"Could not read file at path {path}")