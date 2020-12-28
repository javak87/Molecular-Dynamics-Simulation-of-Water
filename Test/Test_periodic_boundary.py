import numpy as np

import periodic_boundary

def test_apply_boundary_condition():
    """
    This is a test that shows periodic boundary condition in 3D.
    """

    a = np.arange(27).reshape(3,3,3)

    ref = np.array([[[26, 24, 25, 26, 24],
    [20, 18, 19, 20, 18],
    [23, 21, 22, 23, 21],
    [26, 24, 25, 26, 24],
    [20, 18, 19, 20, 18]],

    [[ 8,  6,  7,  8,  6],
    [ 2,  0,  1,  2,  0,],
    [ 5,  3,  4,  5,  3,],
    [ 8,  6,  7,  8,  6],
    [ 2,  0,  1,  2,  0,]],

    [[17,  15,  16, 17, 15],
    [11,  9, 10, 11,  9],
    [14, 12, 13, 14, 12],
    [17, 15, 16, 17, 15],
    [11,  9, 10, 11,  9]],

    [[26, 24, 25, 26, 24],
    [20, 18, 19, 20, 18],
    [23, 21, 22, 23, 21],
    [26, 24, 25, 26, 24],
    [20, 18, 19, 20, 18]],

    [[ 8,  6,  7,  8,  6],
    [ 2,  0,  1,  2,  0],
    [ 5,  3,  4 , 5,  3],
    [ 8,  6,  7,  8,  6],
    [ 2,  0, 1,  2,  0]]])

    # apply periodic boundary condition
    b = periodic_boundary.apply_periodic_boundary(a)

    assert np.all(b==ref), "Boundary condition not periodic."


if __name__=="__main__":
    test_apply_boundary_condition()
