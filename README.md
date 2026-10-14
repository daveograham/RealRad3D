# RealRad3D

**D. Graham & C. Osborne (University of Glasgow)**

**Apache 2.0 License**

## Requirements
Clone https://github.com/Goobley/crtaf-py somewhere and install it to provide the Python helpers

## Installation
Clone the whole RealRad3D folder

In config.py point the TRACER_ROOT path to your own install location
e.g. TRACER_ROOT = "C:/../../RealRad3D/"

## Config
Choose the atom, ion and line index (as used in the atom file) to trace

There are some options in here for the number of wavelength samples, GUI and trace camera size, but be careful :-)

At the moment it is recommended the CAM_HEIGHT does not exceed the current screen resolution

## Starting
Run start_tracer.py with the --path flag and location of your model's DexRT config .yaml
e.g:
python.exe c:/../RealRad3D/start_tracer.py --path "C:/../MyModel/model.yaml"

## Controls
Keyboard arrows or WASD - Pivot the camera

Left mouse - Move the slit position

Right mouse - Move camera target to the closest contact point

F - Return slit to camera centre

H - Return camera focus to model centre and home position

## Saving
P - Writes the current image buffer and spectral window to a pair of files in the model directory
