# 4D Face Reconstruction System from a Single Camera
"""

Made as a final year project for the University of Exeter. The relevant academic paper can be viewed here: https://tinyurl.com/ydewgqjj
This is only to be used for academic purposes.
Please refer to the licensing for more information.

"""

Installation Instructions:
Make sure that your default Python version is 2.7
Make sure you have the latest versions of boost, OpenCV, SciPy, NumPy, Matplotlib, IPython that support Python 2.7 installed

For an easier build of these dependencies, you can try and install conda. After this, a simple "conda install *required dependency*" or "pip install *required dependency*" should do the trick.

The program is currently operational with the following additional libraries:
wxPython (Version 3.0.0.0)
or
QTPy (Version 5.6.2)
Dlib (Version 19.9.0)
Skikit-image (Version 0.13.1)
Mayavi (Version 4.5.0) (Menpo Version)
face-alignment

Make sure to have a morphable model and a shape predictor in the same directory saved as:
MM -- "all_all_all_norm_v2.mat"
SP -- "shape_predictor_68_face_landmarks.dat"

To start this program, make sure that all the files have been extracted and use IPython to boot up MainController.py from the console as follows:

To run, type:
$ iPython MainController.py

In case there is a segmentation fault. Specify the following: $export ETS_TOOLKIT=wx

Disclaimer:
The shape predictor can be acquired for free here:
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

This solution has used the global model from LSFM (https://ibug.doc.ic.ac.uk/resources/lsfm/) as a morphable model. If you'd like to apply for access, please do so through the site above or contact Anastasios Roussos (a.roussos@exeter.ac.uk).
