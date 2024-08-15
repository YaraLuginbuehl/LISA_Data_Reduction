# LISA Data Reduction

This code is part of the Bachelors thesis "Astronomical Spectroscopy with the Long Slit Intermediate Spectrograph for Astronomy (LISA)"
by Yara Luginbühl at the University of Bern, August 2024.

## Description
### main.py
The data taken by this spectrograph is processed with this code resulting in a final spectrum.
Input data : ObjectData (Object frames, flat frames and corresponding darks) and WavelengthCalibrationData (Calibration frames with spectral lines at known wavelengths, flat frames and corresponding darks).
Steps of data reduction (main.py, ObjectData):
1. Creating master frames
2. Definition of sky background
3. Hot pixel detection
4. Image reduction
5. Wavelength calibration
6. Tilt correction
7. Define aperture
8. Final spectrum extraction (Object Spectrum)

### WavelengthCalibration.py
For the wavelength calibration the parameters are calculated seperately.
Steps of Wavelength Calibration (WavelengthCalibration.py,WavelengthCalibrationData)
1. Create master frames
2. Define slit width
3. Hot pixel detection
4. Image reduction
5. Geometric correction
6. Calculation of wavelength calibration matrix
7. Final spectrum extraction (Calibration spectrum)


### Functions.py
All the functions are defined in the seperate python file functions.py and imported into the main from there. They consist of:
1. lin_func
2. gaussian
3. multiple_file_import
4. master_frame
5. OutlierDetection
6. sort

## User Instruction

The code runs for the example data, images taken of Vega at the Observatory of Zimmerwald and the University of Bern. To adjust it one needs to redefine the variables at the beginning of the main.py and the WavelengthCalibration.py documents and save the data the same way that was done here.

### Packages
The following packages are needed:
- General: numpy, math, matplotlib.pyplot, matplotlib.ticker
- Paths: os
- Hot pixel detection: astroscrappy
- Roattion: cv2
- Fits: scipy.optimise

### Data
The data imported in the code is to be saved as follows:
LISA_Data_Reduction
├───main.py
├───WavelengthCalibration.py
├───Functions.py
├───Data
│   ├───ObjectData
│   │   ├───Dark_15s
│   │   ├───Dark_9s
│   │   ├───Flat_9s
│   │   └───Vega_0_5s
│   └───WavelengthCalibrationData
│       ├───Bias
│       ├───Dark_7s
│       ├───Dark_9s
│       ├───Flat
└       └───Neon_new


