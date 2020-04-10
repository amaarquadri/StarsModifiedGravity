import numpy as np
from scipy.interpolate import interp1d
from src.Units import R_sun, state_vector_units


def find_zeros_index(x, round_int=False, find_first=True):
    """
    Calculates the indices where the values in the array are 0.
    Based on this link: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python

    :param x: The array to search for 0's in.
    :param round_int: If False, then x is interpolated to yield a floating point value as the index.
    :param find_first: If True, then only the first zero will be returned, or None if no zeros are found.
    :return: The (optionally floating point) index where the value in the given array equals 0.
    """
    # TODO: verify that this works
    crossings = np.where(np.diff(np.signbit(x)))[0]
    if not round_int:
        crossings = np.array([crossing - x[crossing] / (x[crossing + 1] - x[crossing]) for crossing in crossings])
    return crossings if not find_first else (crossings[0] if len(crossings) > 0 else None)


def interpolate(x, index):
    """
    Linearly interpolates data at floating point indices within the given array or 2D matrix.

    :param x: The array or 2D matrix to interpolate from.
              If x is a matrix, then interpolation is done along the second axis.
    :param index: The (potentially fractional) index to interpolate at.
    :return: The interpolated value or (in the case where x is a matrix) array of values.
    """
    if len(x.shape) == 1:
        # if x is an array
        return np.interp(index, np.arange(len(x)), x)
    else:
        # if x is a 2D matrix
        return interp1d(np.arange(x.shape[1]), x)(index)


def print_state(radius, state, remaining_optical_depth=None):
    normalized_state = state / state_vector_units
    print("Radius: ", radius / R_sun,
          ", Density: ", normalized_state[0],
          ", Temperature: ", normalized_state[1],
          ", Mass: ", normalized_state[2],
          ", Luminosity: ", normalized_state[3],
          ", Optical Depth: ", normalized_state[4],
          ", Remaining Optical Depth: ", remaining_optical_depth if remaining_optical_depth is not None else 'Unknown')


def to_excel(data):
    """
    Converts the given array into a string which can be pasted into an Excel column.
    """
    return ''.join(str(element) + '\n' for element in data)
