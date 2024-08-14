# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:00:23 2024

@author: Yara Luginbühl
"""

import numpy as np
import glob
from astropy.io import fits
import os


def lin_func(a,b,x):
    """
    lin_func : creating a linear trend with parameters a,b.

    Parameters
    ----------
    a : float, slope.
    b : float, offset.
    x : list or numpy array, list of variables.

    Returns
    -------
    list or numpy array, linear trend evaluated for x variables.

    """
    return a*x+b


def gaussian(x, a, b, c):
    """
    gaussian : creating a gaussian curve from the given parameters.
    
    Parameters
    ----------
    x : list or numpy array, array of variables.
    a : float, amplitude (height of peak).
    b : float, position of peak.
    c : float, standard deviation.
    
    Returns
    -------
    list or numpy array, array with the same length as x, gaussian curve evaulated for x variables.

    """
    
    return a * np.exp(-(x - b)**2 / (2 * c**2))




def multiple_file_import(directory, nr_of_files):
    """
    multiple_file_import : importing all fits files in given folder.

    Parameters
    ----------
    directory : str, path to folder.
    nr_of_files : int, number of fits files in folder.

    Returns
    -------
    imported_files : array of str, List of filenames of imported files.
    data : list of 2D numpy arrays, List of imported image data.
    temperature : list of floats or None, list of temperatures of imported files.

    """
    
    if (os.path.isdir(directory) == 0):
        print("The directory is invalid.")
        
    fits_files = []
    imported_files = []
    data = [[] for _ in range(nr_of_files)]
    temperature = [[] for _ in range(nr_of_files)]
    i = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.fit'):
            filepath = os.path.join(directory, filename)
            imported_files.append(filename)
            
            with fits.open(filepath) as hdul:
                fits_files.append(hdul)
                data[i] = hdul[0].data
                
                header = hdul[0].header
                if 'CCD-TEMP' in header:
                    temperature[i] = header['CCD-TEMP']
                else:
                    print('Temperature key not found in FITS header.')
                
                i += 1
                
    return imported_files, data, temperature



def master_frame(data):
    """
    master frame : creating a (mean) master frame from the image data.
    Parameters
    ----------
    data : list of 2D numpy arrays, list with each image.

    Returns
    -------
    master : 2D numpy array, master frame.

    """
    master = np.mean(data, axis = 0)
    
    return master



def OutlierDetection(data, sigma_factor):
    """
    OutlierDetection : detecting outliers in the means of the data
                        recreating original data form without outlier-data.

    Parameters
    ----------
    data : tuple, data as imported by multiple_file_import.
    sigma_factor : float, multiple of standard deviation range defining accepted values.

    Returns
    -------
    name : array of str, list of filenames of accepted files.
    array : list of 2D numpy arrays, list of accepted image data.
    temp : list of floats or None, list of temperatures of accepted files
    
    """

    n = len(data[1])
    mean = []
    for i in range(n):
        mean.append(np.mean(data[1][i]))
    
    data_x = np.arange(1,n+1)
    data_y = mean
    
    median = np.median(data_y)
    standarddeviation = np.abs(data_y - median)
    std_sorted = sorted(standarddeviation)
    
    sigma = std_sorted[int(0.683 * n)]
    
    border_top = median + sigma_factor *sigma
    border_bottom = median - sigma_factor * sigma
    
    indices = np.where((data_y < border_top)&(data_y > border_bottom))
    
    newcols = [[],[]]
    for i in range(len(indices[0])):
        newcols[0].append(data_x[indices[0][i]])
        newcols[1].append(data_y[indices[0][i]])
    
    name =  [[] for _ in range(len(indices[0]))]
    array = [[] for _ in range(len(indices[0]))]
    temp = [[] for _ in range(len(indices[0]))]
    
    for i in range(len(indices[0])):
        name[i] = data[0][indices[0][i]]
        array[i] = data[1][indices[0][i]]
        temp[i] = data[2][indices[0][i]]

    return name, array, temp


def sort(data):
    """
    sort : sorting data imported by multiple_file_import.

    Parameters
    ----------
    data : tuple, data as imported by multiple_file_import.

    Returns
    -------
    data_new1 : array of str, list of filenames of sorted files.
    data_new2 : list of 2D numpy arrays, list of sorted image data.
    data_new3 : list of floats or None, list of temperatures of sorted files
        
    """
    pairs = list(zip(data[0][:], data[1][:], data[2][:]))
    sorted_pairs = sorted(pairs, key=lambda x:int(x[0].split('-')[-1].split('.')[0]))
    
    data_new1, data_new2, data_new3 = zip(*sorted_pairs)
    
    return data_new1, data_new2, data_new3