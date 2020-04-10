import pickle
import time


def get_timestamp():
    """
    Returns a timestamp containing the current date and time.
    Can be used when creating a file name to ensure it is unique.

    :return: A string representation of the current date and time.
    """
    return str(time.strftime("%Y%m%d_%H%M%S"))


def save_star(r_values, state_values, config, file_name=None):
    """
    Saves the detailed structure of the given star.
    The composition of the star and any gravity modifications are saved as well.

    :param r_values: The radius values for the star.
    :param state_values: The star's state matrix.
    :param config: The stellar configuration that was used to generate the data.
    :param file_name: The name of the file to save the stellar structure in.
    """
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'wb') as pout:
        pickle.dump((r_values, state_values, config), pout)


def load_star(file_name=None, return_config=False):
    """
    Loads the detailed structure of the star from the given file.

    :param file_name: The name of the file to read the data from. Should not include the path or file extension.
    :param return_config: If True, then the values for X, Y, Z, lambda_small, and lambda_large are also returned.
    :return: The detailed structure of the the star, and optionally its composition and gravity modifications.
    """
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'rb') as pin:
        r_values, state_values, config = pickle.load(pin)

    print("Read data for " + str(config))
    return (r_values, state_values, config) if return_config else (r_values, state_values)


def save_stellar_data(stellar_data, config, file_name=None):
    if file_name is None:
        file_name = 'stellar_data_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'wb') as pout:
        pickle.dump((stellar_data, config), pout)


def load_stellar_data(file_name=None, return_config=False):
    """
    Loads the aggregated stellar data from the given file.

    :param file_name: The name of the file to read the data from. Should not include the path or file extension.
    :param return_config: If True, then the values for X, Y, Z, lambda_small, and lambda_large are also returned.
    :return: The aggregated stellar data, and optionally the composition and gravity modifications.
    """
    if file_name is None:
        file_name = 'stellar_data_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'rb') as pin:
        stellar_data, config = pickle.load(pin)

    print("Read data for " + str(config))
    return (stellar_data, config) if return_config else stellar_data

