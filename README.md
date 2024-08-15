# LISA Data Reduction

This code is part of the Bachelors thesis "Astronomical Spectroscopy with the Long Slit Intermediate Spectrograph for Astronomy (LISA)"
by Yara Luginbühl at the University of Bern, August 2024.

## Description
### main.py
The data taken by the LISA spectrograph is processed with this code resulting in a final spectrum. <br />
Input data: ObjectData (Object frames, flat frames and corresponding darks) and WavelengthCalibrationData (Calibration frames with spectral lines at known wavelengths, flat frames and corresponding darks).
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
Steps of Wavelength Calibration (WavelengthCalibration.py, WavelengthCalibrationData):
1. Create master frames
2. Define slit width
3. Hot pixel detection
4. Image reduction
5. Geometric correction
6. Calculation of wavelength calibration matrix
7. Final spectrum extraction (Calibration spectrum)


### Functions.py
All the functions are defined in the seperate python file Functions.py and imported into the main and calibration file from there. They consist of:
1. lin_func
2. gaussian
3. multiple_file_import
4. master_frame
5. OutlierDetection
6. sort

## User Instruction

The code runs for the example data, images of Vega.
Example data: Observatory of Zimmerwald, University of Bern, 18.07.2924; ZimMAIN Telescope, Atik 460EX mono camera; Object: Vega
To adjust it one needs to redefine the variables at the beginning of the main.py and the WavelengthCalibration.py documents and save the data the same way that was done here.

### Packages
The following packages are needed:
- General: numpy, math, matplotlib.pyplot, matplotlib.ticker
- Paths: os
- Hot pixel detection: astroscrappy
- Roattion: cv2
- Fits: scipy.optimise

### Data
The data can be downloaded from this link: [Link to Data](https://drive.google.com/drive/folders/1YjgCHpFH25-QFr4tC89KxS_Ab49ypZY8?usp=sharing)
The data imported in the code is to be saved as follows:<br />
LISA_Data_Reduction <br />
├───main.py <br />
├───WavelengthCalibration.py<br />
├───Functions.py<br />
├───Data<br />
&nbsp;&nbsp;&nbsp;&nbsp;   ├───ObjectData<br />
&nbsp;&nbsp;&nbsp;&nbsp;   │   ├───Dark_15s<br />
&nbsp;&nbsp;&nbsp;&nbsp;   │   ├───Dark_9s<br />
&nbsp;&nbsp;&nbsp;&nbsp;   │   ├───Flat_9s<br />
&nbsp;&nbsp;&nbsp;&nbsp;   │   └───Vega_0_5s<br />
&nbsp;&nbsp;&nbsp;&nbsp;   └───WavelengthCalibrationData<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Bias<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Dark_7s<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Dark_9s<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Flat<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      └───Neon_new<br />

### Possible Issues
- One of the most frequently occurring issues were the gaussian fits. Often times the starting parameters (p0 = []) have to be adjusted to be able to generate a fit.
- When there is empty data in the folders the following warning will be generated: WARNING: Unexpected extra padding at the end of the file.  This padding may not be preserved when saving changes. This will however not impact the functionability of the code.
- The detect_cosmics function sometimes detects emission lines which can be adjusted by using an inmask or changing the default parameters. Another option is setting the values that are returned to nan.
