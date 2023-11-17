import numpy as np

def print_dict(dictionary, title=''):
    """Print dictionary formatted."""
    print(title)  # newline
    for k, v in dictionary.items():
        print(f'\t{k:20s}:', end='')
        if isinstance(v, float):
            print(f' {v:06.4f}')
        elif isinstance(v, dict):
            print_dict(v)
        else:
            print(f' {v}')

def lod_mean(list_of_dicts):
    """ Per-key mean of a list of dictionaries. """
    return {k: np.mean([i[k] for i in list_of_dicts]) for k in list_of_dicts[0].keys()}

def keys_append(dictionary, suffix):
    """Appends suffix to all keys in dictionary."""
    return {k + suffix: v for k, v in dictionary.items()}
