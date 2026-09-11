# RealRad3D

**D. Graham & C. Osborne (University of Glasgow)**

**Apache 2.0 License**

## Requirements
Clone https://github.com/Goobley/crtaf-py somewhere and install it to provide the Python helpers

## Installation
Clone the whole RealRad3D folder

In config.py point the paths to your own install location and model folder. ReadRad3D also assumes the model DexRT .yaml config file is stored at the same location

Choose the line to trace at the top. This should correspond to what populations are available in the RT solved output and the atom files (currently Ca II and Mg II but add more as necessary in the RealRad3D path).

There are some options in here for the number of wavelength samples, GUI and trace camera size, but be careful :-)

## Starting
Run start_tracer.py with the --path flag and location of your model's DexRT config .yaml

e.g:

python.exe c:/../RealRad3D/start_tracer.py --path "C:/../MyModel/model.yaml"

## Controls
Keyboard arrows control the pitch and yaw around the model.

Left click places the slit and starts tracing through the wavelength bins

Right click focuses the camera on the first contact point on the model

F - focuses camera look at back on model centre

H - re-sets everything to the home position
