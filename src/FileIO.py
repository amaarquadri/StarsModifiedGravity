import pickle
import time
from src.StellarStructureEquations import X_default, Y_default, Z_default, lambda_small, lambda_large


def get_timestamp():
    return str(time.strftime("%Y%m%d_%H%M%S"))


def save_star(r_values, state_values, file_name=None):
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'wb') as pout:
        pickle.dump((r_values, state_values, X_default, Y_default, Z_default, lambda_small, lambda_large), pout)


def load_star(file_name=None, return_config=False):
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'rb') as pin:
        r_values, state_values, X, Y, Z, lambda_small_value, lambda_large_value = pickle.load(pin)

    print("Read data for X={}, Y={}, Z={}, lambda_small={}, lambda_large={}".format(X, Y, Z, lambda_small_value,
                                                                                    lambda_large_value))
    return (r_values, state_values, X, Y, Z, lambda_small_value, lambda_large_value) if return_config else \
        (r_values, state_values)


def save_stellar_data(stellar_data, file_name=None):
    if file_name is None:
        file_name = 'stellar_data_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'wb') as pout:
        pickle.dump((stellar_data, X_default, Y_default, Z_default, lambda_small, lambda_large), pout)


def load_stellar_data(file_name=None, return_config=False):
    if file_name is None:
        file_name = 'stellar_data_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'rb') as pin:
        stellar_data, X, Y, Z, lambda_small_value, lambda_large_value = pickle.load(pin)

    print("Read data for X={}, Y={}, Z={}, lambda_small={}, lambda_large={}".format(X, Y, Z, lambda_small_value,
                                                                                    lambda_large_value))
    return (stellar_data, X, Y, Z, lambda_small_value, lambda_large_value) if return_config else stellar_data
