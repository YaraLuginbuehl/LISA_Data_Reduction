# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:16:27 2024

@author: Yara Luginbühl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
import astroscrappy
import os

from Functions import lin_func, multiple_file_import, master_frame, gaussian, sort, OutlierDetection


# The following code is a wavelength calibration for the LISA spectrograph. 
# It entails the following steps:
    # Import files: Calibration, Dark (Exp: Calibration), Flat, Dark (Exp: Flat) 
    # Sort files and detect outliers
    # Create master frames
    # Define slitwidth
    # Hot pixel detection
    # Image reduction
    # Geometric correction
    # Calculation of wavelength calibration matrix
    # Calibration spectrum extraction
    
"""
Results
-------
Lambda : 2D numpy array, the new grid on which the image has to be interpolated for the wavelength calibration.
Cal_image_mean : 1D numpy array, the final spectrum of the calibration frame.

"""
#%% DIRECTORY

# Defining working directory: 
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir)


# Defining path to data:
WavelengthCalibrationData_directory = os.path.join("Data","WavelengthCalibrationData")


#%% DEFINITION OF VARIABLES

# GENERAL VARIABLES
# xlen : int, length of x axis [px].
# ylen : int, length of y axis [px].
ylen = 2199
xlen = 2749

y = np.arange(1,ylen+1)
x = np.arange(1,xlen+1)

# DEFINITION OF SLITWIDTH
# brightest_line_left : int, left border of brightest line.
# brighetst_line_right : int, right border of brightest line.
brightest_line_left = 885
brightest_line_right = 905

# HOT PIXEL DETECTION
# inmask_coord : list, list of left and right borders of lines used as inmask for hot pixel detection.
inmask_coord = [(885, 925), (935, 975), (985, 1220), (1240, 1287),
                 (1285, 1325), (1338, 1390), (1475, 1510), (1530, 1570), (1605, 1690)]

# WAVELENGTH CALIBRATION
# line_borders : list, list of left and right borders of lines used for wavelength calibration [px].
# wavelengths : list, wavelengths corresponding to lines defined in line_borders [nm or other].
line_borders = [(890, 912), (906,922), (936, 957), (983,1006), (1045,1080),(1085,1112), (1112,1140), (1185,1220), (1290, 1330),(1475, 1510),(1530,1570)]
wavelengths = [585.249, 588.190, 594.483, 603.000, 614.306, 621.728, 626.650, 640.225, 659.298, 692.947, 703.241]



#%% IMPORT DATA

Flat = multiple_file_import(directory = os.path.join(WavelengthCalibrationData_directory, "Flat"), nr_of_files = 100)
Dark_Cal = multiple_file_import(directory = os.path.join(WavelengthCalibrationData_directory, "Dark_7s"), nr_of_files = 100)
Dark_Flat = multiple_file_import(directory = os.path.join(WavelengthCalibrationData_directory, "Dark_9s"), nr_of_files = 100)
Cal = multiple_file_import(directory = os.path.join(WavelengthCalibrationData_directory, "Neon_new"), nr_of_files = 100)


#%% SORT DATA & extract data without outliers

Flat = sort(Flat)
Dark_Cal = sort(Dark_Cal)
Dark_Flat = sort(Dark_Flat)
Cal = sort(Cal)

Flat = OutlierDetection(Flat, 1.5)
Dark_Flat = OutlierDetection(Dark_Flat, 1.5)
Dark_Cal = OutlierDetection(Dark_Cal, 1.5)
Cal = OutlierDetection(Cal, 1.5)


#%% MASTER FRAMES

Dark_Cal_m = master_frame(Dark_Cal[1])
Dark_Flat_m = master_frame(Dark_Flat[1])
Flat_m = master_frame(Flat[1])
Cal_m = master_frame(Cal[1])


#%%  DEFINE SLITHEIGHT
# Slitwidth is defined as the FWHM of the gaussian fit of the brightest line.

# Defining the curve to be fit:
slit_image = np.zeros(ylen)
for i in range(ylen):
    slit_image[i] = np.mean(Cal_m[i][brightest_line_left:brightest_line_right])

paramsbild, covariancebild = curve_fit(gaussian, y, slit_image, p0=[
                                       np.max(slit_image), np.argmax(slit_image), np.std(slit_image)])

FWHMbild = 2 * np.sqrt(2*np.log(2))*abs(paramsbild[2])
b = int(paramsbild[1] + FWHMbild/2)
t = int(paramsbild[1] - FWHMbild/2)

# Defining the final slitwidth and its error:
slitwidth = int(b - t)
sigma_slitwidth = 2 * np.sqrt(2*np.log(2))*np.sqrt(covariancebild[2,2])


# Cropping the images:
Flat_m = Flat_m[t:b, :]
Dark_Cal_m = Dark_Cal_m[t:b, :]
Dark_Flat_m = Dark_Flat_m[t:b, :]
Cal_m = Cal_m[t:b, :]
y = y[t:b]


#%% HOT PIXEL DETECTION
# Hot pixels are detected with default values outside of inmask and with defined values in inmask

inmask = np.zeros_like(Cal_m, dtype=bool)
opp_inmask = np.full(Cal_m.shape, True)

for start, end in inmask_coord:
    inmask[:, start:end] = True
    opp_inmask[:, start:end] = False

hot_px_dark_cal, Dark_Cal_m = astroscrappy.detect_cosmics(Dark_Cal_m)
hot_px_dark_9s, Dark_Flat_m = astroscrappy.detect_cosmics(Dark_Flat_m)
hot_px_Cal1, Cal_m1 = astroscrappy.detect_cosmics(Cal_m, inmask=inmask)
hot_px_Cal, Cal_m = astroscrappy.detect_cosmics(Cal_m1, sigfrac = 6.6, inmask = opp_inmask)
hot_px_Flat, Flat_m = astroscrappy.detect_cosmics(Flat_m)


#%% IMAGE CALIBRATION

# Flat normalisation:
Flat_m = Flat_m - Dark_Flat_m
Flat_norm = Flat_m/np.amax(Flat_m)

# Final Calibration:
Cal_corr = (Cal_m - Dark_Cal_m)/Flat_norm


#%% WAVELENGTH CALIBRATION
# Creating a gaussian fit of each line for each y-height in the slitwidth,
# Every 100 steps the line borders are moved since there is a tilt in the lines. 

line_nr = len(line_borders)
b_lines = np.zeros((slitwidth, line_nr))

for i in range(slitwidth):
    for j in range(line_nr):

        left, right = line_borders[j]
        params, covariance = curve_fit(gaussian, x[left:right], Cal_corr[i, left:right], p0=[
                                           np.max(Cal_corr[i, left:right]), left + (right-left)/2, (right-left)/2])
        a, b_lines[i][j], c = params
        fit = gaussian(x[left:right], a, b_lines[i][j], c)

    if i % 100 == 0 and i != 0:
        line_borders = [(start+1, end+1) for start, end in line_borders]
    
# Creating a linear fit of b parameters with wavelengths for each y-height:
wavelength_fit = np.zeros((slitwidth, line_nr))
Lambda = np.zeros((slitwidth,xlen))

for i in range(slitwidth):
    var, cov = curve_fit(lin_func, b_lines[i,:],wavelengths)
    err = np.sqrt(np.diag(cov))
    wavelength_fit[i] = var[0] + var[1]*b_lines[i,:]
    Lambda[i,:] = var[0] + var[1]*x

# Defining the wavelength axis based on the wavelength grid:
lambda_min = int(np.mean(Lambda[:,0]))
lambda_max = int(np.mean(Lambda[:,-1]))

lambda_target = np.linspace(lambda_min, lambda_max, (2*xlen))

# Interpolating the calibration frame on the new wavelength grid:
Cal_image = np.zeros((slitwidth, 2*xlen))
for i in range(slitwidth):
    Cal_image[i,:] = np.interp(lambda_target, Lambda[i,:],Cal_corr[i,:])


#%% SPECTRUM EXTRACTION
# Taking the mean over the slitwidth to extract final spectrum (Cal_image_mean)

Cal_image_mean = np.zeros(2*xlen)

for i in range(2*xlen):
    Cal_image_mean[i] = np.mean(Cal_image[:, i])

# Creating a plot of the final spectra:
plt.rcParams['font.size'] = 30
fig, ax = plt.subplots()
ax.plot(lambda_target, Cal_image_mean, "k")
plt.title("Calibration Spectrum")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [arb. units]")
ax.xaxis.set_major_locator(MultipleLocator(50))

