# LISA Data Reduction

This code is part of the Bachelors thesis "Astronomical Spectroscopy with the Long Slit Intermediate Spectrograph for Astronomy (LISA)"
by Yara Luginbühl at the University of Bern, August 2024.

## Description
The code is visualised in the following image: ![main_visualisation](https://github.com/user-attachments/assets/25ba03fe-7c21-42b7-b091-6481ae08dc98)


### main.py
The data taken by the LISA spectrograph is processed with this code resulting in a final spectrum. <br />
Input data: ObjectData (Object frames, flat frames and corresponding darks) and WavelengthCalibrationData (Calibration frames with spectral lines at known wavelengths, flat frames and corresponding darks).<br />
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
For the wavelength calibration the parameters are calculated seperately.<br />
Steps of Wavelength Calibration (WavelengthCalibration.py, WavelengthCalibrationData):
1. Create master frames
2. Define slit width
3. Hot pixel detection
4. Image reduction
5. Geometric correction
6. Calculation of wavelength calibration matrix
7. Final spectrum extraction (Calibration spectrum)

### main_withplots.py
This code is the same as the main.py with some additional steps: There are five additional plots created to visualise the relevant steps. If the observed object is a star (object_is_blackbody = True), there is an additional analysis which calculates the blackbody spectrum (Planck's law) and the position of maximum (Wien's law) needing the tempreature as a variable. There is one additional package imported (matplotlib.patches).

### Functions.py
All the functions are defined in the seperate python file Functions.py and imported into the main and calibration file from there. They consist of:
1. lin_func
2. gaussian
3. multiple_file_import
4. master_frame
5. OutlierDetection
6. sort
7. black_body_radiation

## User Instruction

The code runs for the example data, images of Vega.
Example data: Observatory of Zimmerwald, University of Bern, 18.07.2024; ZimMAIN Telescope, Atik 460EX mono camera; Object: Vega <br />
To adjust it one needs to redefine the variables at the beginning of the main.py (or main_withplots.py) and the WavelengthCalibration.py documents and save the data the same way that was done here.

### Packages
The following packages are needed:
- General: numpy, math, matplotlib.pyplot, matplotlib.ticker, matplotlib.patches
- Paths: os
- Hot pixel detection: astroscrappy
- Rotation: cv2
- Fits: scipy.optimise
- black_body_radiation: scipy.constants

### Data
The example data can be downloaded from this link: [Link to Data](https://drive.google.com/drive/folders/1YjgCHpFH25-QFr4tC89KxS_Ab49ypZY8?usp=sharing)<br />
In the ObjectData folder all images acquired during an observation are saved. Here Vega_0_5s are the object images. Flat_9s, Dark_9s, Dark_15s are the frames used for the image reduction which are named with the corresponding exposure time to avoid a mix-up. The same scheme was followed in the WavelengthCalibrationData folder: Neon_new are the calibration images and Flat, Dark_7s, Dark_9s the frames for the image reduction. <br />
<br />
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Dark_7s<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Dark_9s<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      ├───Flat<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      └───Neon_new<br />

If new data is acquired it can be used in the code as follows: All the data from the observation (object frames, flat frames, dark (object), dark (flat)) are to be saved in individual folders in the ObjectData folder. Similarly, the data used for the wavelength calibration is saved in the WavelengthCalibrationData folder. If the folders containing the images have new names, this can be solved in the code: In the subsection "IMPORT DATA" in main.py (or main_withplots.py) and WavelengthCalibration.py one can simply change the variables of the mulitple_file_import function: In the directory one has to change the folder name and the nr_of_files has to be changed to the new number of files in each folder.<br />

### Possible Issues
- One of the most frequently occurring issues were the gaussian fits. Often times the starting parameters (p0 = [...]) have to be adjusted to be able to generate a fit.
- When there is empty data in the folders the following warning will be generated: WARNING: Unexpected extra padding at the end of the file.  This padding may not be preserved when saving changes. This will however not impact the functionability of the code.
- The detect_cosmics function sometimes detects emission lines which can be adjusted by using an inmask or changing the default parameters. Another option is setting the values that are returned to nan.
