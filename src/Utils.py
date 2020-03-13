import numpy as np


def find_zeros(x, round_int=False, find_first=True):
    """
    Calculates the indices where the values in the array are 0.
    Based on this link: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python

    :param x: The array to search for 0's in.
    :param round_int: If False, then the index is adjusted.
    :param find_first: If True, then only the first zero will be returned, or None if no zeros are found.
    :return:
    """
    # TODO: verify that this works
    crossings = np.where(np.diff(np.signbit(x)))[0]
    if not round_int:
        crossings = np.array([crossing - x[crossing] / (x[crossing + 1] - x[crossing]) for crossing in crossings])
    return crossings if not find_first else (crossings[0] if len(crossings) > 0 else None)


def interpolate(x, index):
    """
    Linearly interpolates data at fractional indices within the given array.
    """
    return np.interp(index, np.arange(len(x)), x)
