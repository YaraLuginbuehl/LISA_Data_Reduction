# LISA Data Reduction

This code is part of the Bachelors thesis "Astronomical Spectroscopy with the Long Slit Intermediate Spectrograph for Astronomy (LISA)"
by Yara Luginbühl at the University of Bern, August 2024.

## Description
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

For the wavelength calibration the parameters are calculated seperately.
Steps of Wavelength Calibration (WavelengthCalibration.py,WavelengthCalibrationData)
1. Create master frames
2. Define slit width
3. Hot pixel detection
4. Image reduction
5. Geometric correction
6. Calculation of wavelength calibration matrix
7. Final spectrum extraction (Calibration spectrum)
