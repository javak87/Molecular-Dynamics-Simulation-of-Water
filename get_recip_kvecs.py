import numpy as np
from math import pi, floor

def get_recip_kvecs(k_cut, box_len):
        """
        Get reciprocal lattice vectors k-vecs in the upper sphere and 
        k-vecs in the below part can be mirrored by the upper ones, 
        where k-vecs in the xy plane have been considered for following calculation.
        """
        k_vec = np.empty((0,3), int)
        radius = k_cut * box_len / (2 * pi)
        radius_i = floor(radius)
        for i in range(0, radius_i+1):
            for j in range(0, radius_i+1):
                for k in range(0, radius_i+1):
                    if (( i * i + j * j + k * k)<= radius ** 2):
                        k_vec = np.append(k_vec,np.array([[i,j,k]]),axis=0)
        k_vec_x = np.asarray(np.matmul(k_vec, np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        k_vec_y = np.asarray(np.matmul(k_vec, np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])))
        k_vec_xy = np.asarray(np.matmul(k_vec, np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])))
        k_vec = np.vstack((k_vec, k_vec_x, k_vec_y, k_vec_xy)) 
        k_vec = k_vec[~np.all(k_vec == 0, axis=1)]
        k_vec = np.unique(k_vec, axis=0)
        k_vec_in_xy = k_vec[k_vec[:,2] == 0] * 2 * pi/box_len
        k_vec = k_vec * 2 * pi/box_len
        k_square_in_xy = np.sum(k_vec_in_xy ** 2, 1)
        k_square = np.sum(k_vec ** 2, 1)
        return k_vec_in_xy, k_vec, k_square_in_xy, k_square
