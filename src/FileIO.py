import pickle
import time


def get_timestamp():
    return str(time.strftime("%Y%m%d_%H%M%S"))


def save_star(r_values, state_values, file_name=None):
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'wb') as pout:
        pickle.dump((r_values, state_values), pout)


def load_star(file_name=None):
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'rb') as pin:
        r_values, state_values = pickle.load(pin)
    return r_values, state_values


def save_stellar_data(stellar_data, file_name=None):
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'wb') as pout:
        pickle.dump(stellar_data, pout)


def load_stellar_data(file_name=None):
    if file_name is None:
        file_name = 'star_pickle_' + get_timestamp()
    with open('../Stellar Data/' + file_name + '.pickle', 'rb') as pin:
        stellar_data = pickle.load(pin)
    return stellar_data
