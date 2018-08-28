import numpy as np

def input_check_Nx3x3(x):
	"""
	Check x to be of dimension Nx3x3
	Jacob Niehus
	"""
	theSize = np.shape(x)
	N = 1
	if(len(theSize) > 2):
		# 1. Input must be of size N x 3
		if (3, 3) not in (theSize[:2], theSize[-2:]):
			raise ValueError('Not a N x 3 x 3 array')
		# 2. Make it into a Nx3x3 array
		if (theSize[1:] != (3, 3)):
			x = np.rollaxis(x, -1)
		N = x.shape[0]
		# 3. If N == 2, make it into a 2-D array
		if (x.shape[0] == 1):
			x = x[0]
	elif(theSize != (3, 3)):
		raise ValueError('Not a 3 x 3 array')
	return x, N
def input_check_Nx1(x):
    """
    Check x to be of dimension Nx1 and reshape it as a 1-D array
    Adhika Lie
    """
    x = np.atleast_1d(x)
    theSize = np.shape(x)

    if(len(theSize) > 1):
        # 1. Input must be of size N x 1
        if ((theSize[0] != 1) & (theSize[1] != 1)):
            raise ValueError('Not an N x 1 array')
        # 2. Make it into a 1-D array
        x = x.reshape(np.size(x))
    elif (theSize[0] == 1):
        x = x[0]

    return x, np.size(x)

def dcm2angle(C, output_unit='rad', rotation_sequence='ZYX'):
	"""
	This function converts a Direction Cosine Matrix (DCM) into the three
	rotation angles.
	The DCM is described by three sucessive rotation rotAngle1, rotAngle2, and
	rotAngle3 about the axis described by the rotation_sequence.
	The default rotation_sequence='ZYX' is the aerospace sequence and rotAngle1
	is the yaw angle, rotAngle2 is the pitch angle, and rotAngle3 is the roll
	angle. In this case DCM transforms a vector from the locally level
	coordinate frame (i.e. the NED frame) to the body frame.
	This function can batch process a series of rotations (e.g., time series
	of direction cosine matrices).
	Parameters
	----------
	C : {(3,3), (N,3,3), or (3,3,N)}
		direction consine matrix that rotates the vector from the first frame
		to the second frame according to the specified rotation_sequence.
	output_unit : {'rad', 'deg'}, optional
			Rotation angles. Default is 'rad'.
	rotation_sequence : {'ZYX'}, optional
			Rotation sequences. Default is 'ZYX'.
	Returns
	-------
	rotAngle1, rotAngle2, rotAngle3 :  angles
			They are a sequence of angles about successive axes described by
			rotation_sequence.
		Notes
	-----
	The returned rotAngle1 and 3 will be between   +/- 180 deg (+/- pi rad).
	In contrast, rotAngle2 will be in the interval +/- 90 deg (+/- pi/2 rad).
	In the 'ZYX' or '321' aerospace sequence, that means the pitch angle
	returned will always be inside the closed interval +/- 90 deg (+/- pi/2 rad).
	Applications where pitch angles near or larger than 90 degrees in magnitude
	are expected should used alternate attitude parameterizations like
	quaternions.
	"""
	C, N = input_check_Nx3x3(C)
	if(rotation_sequence == 'ZYX'):
		rotAngle1 = np.arctan2(C[..., 0, 1], C[..., 0, 0])   # Yaw
		rotAngle2 = -np.arcsin(C[..., 0, 2])  # Pitch
		rotAngle3 = np.arctan2(C[..., 1, 2], C[..., 2, 2])  # Roll
	else:
		raise NotImplementedError('Rotation sequences other than ZYX are not currently implemented')
	if(output_unit == 'deg'):
		rotAngle1 = np.rad2deg(rotAngle1)
		rotAngle2 = np.rad2deg(rotAngle2)
		rotAngle3 = np.rad2deg(rotAngle3)
	return rotAngle1, rotAngle2, rotAngle3
	
def angle2dcm(rotAngle1, rotAngle2, rotAngle3, input_unit='rad',
              rotation_sequence='ZYX', output_type='ndarray'):
    """
    This function converts Euler Angle into Direction Cosine Matrix (DCM).
    The DCM is described by three sucessive rotation rotAngle1, rotAngle2, and
    rotAngle3 about the axis described by the rotation_sequence.
    The default rotation_sequence='ZYX' is the aerospace sequence and rotAngle1
    is the yaw angle, rotAngle2 is the pitch angle, and rotAngle3 is the roll
    angle. In this case DCM transforms a vector from the locally level
    coordinate frame (i.e. the NED frame) to the body frame.
    This function can batch process a series of rotations (e.g., time series
    of Euler angles).
    Parameters
    ----------
    rotAngle1, rotAngle2, rotAngle3 : angles {(N,), (N,1), or (1,N)}
            They are a sequence of angles about successive axes described by
            rotation_sequence.
    input_unit : {'rad', 'deg'}, optional
            Rotation angles. Default is 'rad'.
    rotation_sequence : {'ZYX'}, optional
            Rotation sequences. Default is 'ZYX'.
    output_type : {'ndarray','matrix'}, optional
            Output type. Default is 'ndarray'.
    Returns
    --------
    C : {3x3} Direction Cosine Matrix
    Notes
    -----
    Programmer:    Adhika Lie
    Created:    	 May 03, 2011
    Last Modified: January 12, 2016
    """
    rotAngle1, N1 = input_check_Nx1(rotAngle1)
    rotAngle2, N2 = input_check_Nx1(rotAngle2)
    rotAngle3, N3 = input_check_Nx1(rotAngle3)

    if(N1 != N2 or N1 != N3):
        raise ValueError('Inputs are not of same dimensions')
    if(N1 > 1 and output_type != 'ndarray'):
        raise ValueError('Matrix output requires scalar inputs')

    R3 = np.zeros((N1, 3, 3))
    R2 = np.zeros((N1, 3, 3))
    R1 = np.zeros((N1, 3, 3))

    if(input_unit == 'deg'):
        rotAngle1 = np.deg2rad(rotAngle1)
        rotAngle2 = np.deg2rad(rotAngle2)
        rotAngle3 = np.deg2rad(rotAngle3)

    R3[:, 2, 2] = 1.0
    R3[:, 0, 0] = np.cos(rotAngle1)
    R3[:, 0, 1] = np.sin(rotAngle1)
    R3[:, 1, 0] = -np.sin(rotAngle1)
    R3[:, 1, 1] = np.cos(rotAngle1)

    R2[:, 1, 1] = 1.0
    R2[:, 0, 0] = np.cos(rotAngle2)
    R2[:, 0, 2] = -np.sin(rotAngle2)
    R2[:, 2, 0] = np.sin(rotAngle2)
    R2[:, 2, 2] = np.cos(rotAngle2)

    R1[:, 0, 0] = 1.0
    R1[:, 1, 1] = np.cos(rotAngle3)
    R1[:, 1, 2] = np.sin(rotAngle3)
    R1[:, 2, 1] = -np.sin(rotAngle3)
    R1[:, 2, 2] = np.cos(rotAngle3)

    if rotation_sequence == 'ZYX':
        try:
            # Equivalent to C = R1.dot(R2.dot(R3)) for each of N inputs but
            # implemented efficiently in C extension
            C = np.einsum('nij, njk, nkm -> nim', R1, R2, R3)
        except AttributeError:
            # Older NumPy without einsum
            C = np.zeros((N1, 3, 3))
            for i, (R1, R2, R3) in enumerate(zip(R1, R2, R3)):
                C[i] = R1.dot(R2.dot(R3))
    else:
        raise NotImplementedError('Rotation sequences other than ZYX are not currently implemented')

    if(N1 == 1):
        C = C[0]
    if(output_type == 'matrix'):
        C = np.matrix(C)

    return C