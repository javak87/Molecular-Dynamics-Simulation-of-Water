import numpy as np

def apply_periodic_boundary(a: np.ndarray):
    """ Applies a perodic boundary condition to a 
        ghost cell layer in three spatial dimensions.
    """
    a = np.pad(a, (1,1), 'wrap')
    return a
