# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:18:59 2024

@author: Yara Luginbühl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cv2
import math as m
from scipy.optimize import curve_fit
import astroscrappy
import os

from Functions import multiple_file_import, master_frame, gaussian, sort, OutlierDetection
from WavelengthCalibration import Lambda, t as t_lambda


# The following code is a data reduction pipeline for the LISA, Sheylak Instruments. 
# It entails the following steps:
    # Import files: Object, Dark (Exp: Object), Flat, Dark (Exp: Flat) 
    # Sort files, detect outliers and choose files to use
    # Create master frames and define sky background
    # Hot pixel detection
    # Image reduction
    # Wavelength calibration
    # Tilt correction
    # Define aperture
    # Final spectrum extraction

"""
Results
-------
From WavelengthCalibration.py : Plot of final calibration spectrum.
Object_corr_mean : 1D numpy array, final dark, flat and sky background corrected spectrum
Object_alt_mean : 1D numpy array, final dark and sky background corrected spectrum

"""
#%% DIRECTORY

# Defining working directory: 
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir)

# Defining directory to data: 
ObjectData_directory = os.path.join("Data","ObjectData")

#%% DEFINITION OF VARIABLES

# GENERAL VARIABLES
# xlen : int, length of x axis [px].
# ylen : int, length of y axis [px].
ylen = 2199
xlen = 2749

y = np.arange(1, ylen+1)
x = np.arange(1, xlen+1)

# first, last : int, choice of which images to use (index).
first = 0 
last = -1

# top, bottom, left, right : int, approximate position of spectrum.
top = 1150
bottom = 1225
left = 0
right = 1500

slitheight = bottom - top

# exp_time_object : int, exposuret time of object frames [seconds]
# exp_time_dark : int, exposuret time of dark frames [seconds]
exp_time_object = 0.5
exp_time_dark = 15

# SKY BACKGROUND
# SB_t : int, upper bound of sky background cutout.
SB_t = 1000


#%% IMPORT FILES

Object =  multiple_file_import(directory = os.path.join(ObjectData_directory,"Vega_0_5s"),nr_of_files = 10)
Dark_obj =  multiple_file_import(directory = os.path.join(ObjectData_directory,"Dark_15s"),nr_of_files = 20)
Flat = multiple_file_import(directory = os.path.join(ObjectData_directory,"Flat_9s"),nr_of_files = 100)
Dark_flat = multiple_file_import(directory = os.path.join(ObjectData_directory,"Dark_9s"),nr_of_files = 100)


#%% SORT AND CHOOSE FILES

Object = sort(Object)
Dark_obj = sort(Dark_obj)
Flat = sort(Flat)
Dark_flat = sort(Dark_flat)

Object = Object[0][first:last],Object[1][first:last], Object[2][first:last]
Dark_obj = Dark_obj[0][14:], Dark_obj[1][14:], Dark_obj[2][14:]

Flat = OutlierDetection(Flat, 1.5)
Dark_flat = OutlierDetection(Dark_flat, 1.5)


#%% MASTER FRAMES, DEFINITION OF SKY BACKGROUND

Object_m = master_frame(Object[1])
Dark_obj_m = master_frame(Dark_obj[1])
Flat_m = master_frame(Flat[1])
Dark_flat_m = master_frame(Dark_flat[1])

SB = Object_m[SB_t:SB_t + slitheight]
Flat_SB = Flat_m[SB_t:SB_t + slitheight]
Dark_obj_SB = Dark_obj_m[SB_t:SB_t + slitheight]
Dark_flat_SB = Dark_flat_m[SB_t:SB_t + slitheight]

# Defining cutout of image based on approximated position:
Object_m = Object_m[top:bottom]
Dark_obj_m = Dark_obj_m[top:bottom]
Flat_m = Flat_m[top:bottom]
Dark_flat_m = Dark_flat_m[top:bottom]

y = y[top:bottom]


#%% HOT PIXEL DETECTION

hot_px_Object, Object_m = astroscrappy.detect_cosmics(Object_m)
hot_px_Dark_obj, Dark_obj_m = astroscrappy.detect_cosmics(Dark_obj_m)
hot_px_Flat, Flat_m = astroscrappy.detect_cosmics(Flat_m)
hot_px_Dark_flat, Dark_flat_m = astroscrappy.detect_cosmics(Dark_flat_m)

hot_px_SB, SB = astroscrappy.detect_cosmics(SB)
hot_px_Flat_SB, Flat_SB = astroscrappy.detect_cosmics(Flat_SB)
hot_px_Dark_obj_SB, Dark_obj_SB = astroscrappy.detect_cosmics(Dark_obj_SB)
hot_px_Dark9_SB, Dark_flat_SB = astroscrappy.detect_cosmics(Dark_flat_SB)


#%% IMAGE REDUCTION

Dark_obj_m = (exp_time_object/exp_time_dark) * Dark_obj_m
Dark_obj_SB = (exp_time_object/exp_time_dark) * Dark_obj_SB

# Flat normalisation:
Flat_norm = (Flat_m - Dark_flat_m)/np.amax(Flat_m-Dark_flat_m)
Flat_SB_norm = (Flat_SB - Dark_flat_SB)/np.amax(Flat_SB - Dark_flat_SB)

# Correcting for dark and flat:
Object_corr = (Object_m - Dark_obj_m)/Flat_norm
SB_corr = (SB - Dark_obj_SB)/Flat_SB_norm

# Correcting for sky background:
Object_corr = Object_corr - SB_corr

# Alternative solution: flat-uncorrected:
Object_alt = (Object_m - Dark_obj_m) - (SB - Dark_obj_SB)


#%% WAVELENGTH CALIBRATION
# The image is interpolated on the new wavelength grid calculated by WavelengthCalibration.py

# Defining the new wavelength axis lambda_target from the wavelength grid:
lambda_min = int(np.mean(Lambda[:,0]))
lambda_max = int(np.mean(Lambda[:,-1]))
lambda_target = np.linspace(lambda_min, lambda_max, (2*xlen))

Object_corr_image = np.zeros((slitheight, 2*xlen))
Object_alt_image = np.zeros((slitheight, 2*xlen))
for i in range(slitheight):
    Object_corr_image[i,:] = np.interp(lambda_target, Lambda[(t_lambda-top)+i,:xlen],Object_corr[i,:])
    Object_alt_image[i,:] = np.interp(lambda_target, Lambda[(t_lambda-top)+i,:xlen],Object_alt[i,:])


#%% TILT CORRECTION
# A gaussian fit is done for each pixel in the x-length. 

# Creating a gaussian fit for the values at each x pixel:
x_new = np.linspace(1,2749, 2*2749)
xlen_new = len(x_new)
line = np.zeros((2*right-2*left, 3))

for i in range(2*right-2*left):
    params, covariance = curve_fit(gaussian, y, Object_corr_image[:, i], p0=[
                                    np.max(Object_corr_image[:,i]), top + (top-bottom)/2, (top-bottom)/2])
    line[i][2], line[i][0], line[i][1] = params

# Creating a linear fit of the b parameters and calculating angle to turn:
n, q = np.polyfit(x_new[2*left:2*right], line[:, 0], deg=1)
line_fit = n*x_new[2*left:2*right]+q

angle = m.degrees(np.arctan((line_fit[0]-line_fit[-1])/len(x_new[0:(left-1)])))

# Calculating the rotation matrix:
M = cv2.getRotationMatrix2D((xlen_new/2, slitheight/2), -angle, 1)

# Performing the rotation:
Object_alt_image = cv2.warpAffine(Object_alt_image, M, (xlen_new, slitheight))
Object_corr_image = cv2.warpAffine(Object_corr_image, M, (xlen_new, slitheight))


#%% OBJECT APERTURE
# Aperture is defined as the FWHM of the gaussian fit when taking the mean for each y pixel

# Taking the mean for each y pixel:
aperture_image = np.zeros(slitheight)
for i in range(slitheight):
    aperture_image[i] = np.mean(Object_corr_image[i][:])

paramsbild, covariancebild = curve_fit(gaussian, y, aperture_image, p0=[
                                        np.max(aperture_image), top + (top-bottom)/2, (top-bottom)/2])
FWHMbild = 2 * np.sqrt(2*np.log(2))*abs(paramsbild[2])
b = int(paramsbild[1] + FWHMbild/2)
t = int(paramsbild[1] - FWHMbild/2)

aperture = int(b - t)
sigma_aperture = 2 * np.sqrt(2*np.log(2))*np.sqrt(covariancebild[2,2])

# Cropping images to aperture:
Object_corr_image = Object_corr_image[t-top:b-top]
Object_alt_image = Object_alt_image[t-top:b-top]


#%% SPECTRUM EXTRACTION
# Taking the mean for each x pixel returns the final spectra

Object_corr_mean = np.zeros(xlen_new)
Object_alt_mean = np.zeros(xlen_new)

for i in range(xlen_new):
    Object_corr_mean[i] = np.mean(Object_corr_image[:, i])
    Object_alt_mean[i] = np.mean(Object_alt_image[:,i])
    
# Dividing final spectra by exposure time: 
Object_corr_mean = Object_corr_mean/exp_time_object
Object_alt_mean = Object_alt_mean/exp_time_object

# Creating plots of the final spectra:
fig, ax = plt.subplots()
ax.plot(lambda_target,Object_corr_mean, "k")
plt.title("Vega Spectrum (Dark+SB)")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [arb. units]")
ax.xaxis.set_major_locator(MultipleLocator(50))

fig, ax = plt.subplots()
ax.plot(lambda_target,Object_alt_mean, "k")
plt.title("Vega Spectrum (Dark+Flat+SB)")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [arb. units]")
ax.xaxis.set_major_locator(MultipleLocator(50))




