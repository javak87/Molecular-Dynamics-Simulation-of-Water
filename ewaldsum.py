import numpy as np
from numpy import inf 
import scipy.constants as constants
from lattice_config import LatticeConfig
from math import pi, sqrt, floor

class EwaldSummation:
    """
        Returns Long Range part of potential energy using ewald summation
        E^L = 1/(2V epsilon_0) sum_{k<k_cut}{|S(k)|^2} (e^{-sigma^2 k^2/2})/(k^2)
        S(k) = sum_{a=1}^{N} q_a e^{i <k, r_a>}
        """
    def __init__(self, pos, charges, box_len, r_cut=None, k_cut=None,
                 acc_p=12.0):
        
        self._pos = pos
        self._charges = charges
        self._box_len = box_len
        self._vol = box_len ** 3
        self._acc_p = acc_p
        self._rmax = r_cut if r_cut \
            else box_len
        self._kmax = k_cut if k_cut \
            else 2 * acc_p / self._rmax
        self._sigma = self._rmax / sqrt(2 * self._acc_p)
    
    def _get_points_in_sphere(self):
        radius = floor(self._kmax * self._box_len / (2 * pi))
        k_vec = np.empty((0,3), int)
        for i in range(radius+1):
            for j in range(radius+1):
                for k in range(radius+1):
                    if (( i * i + j * j + k * k)<= radius ** 2):
                        k_vec = np.append(k_vec,np.array([[i,j,k]]),axis=0)
        k_vec = k_vec * 2 * pi/self._box_len
        return k_vec
    
    def _calc_longrange(self):
        """
        Calculate pre parameters 
        """
        epsilon = constants.epsilon_0
        volume = self._vol
        sigma = self._sigma
        prefac = 1/(volume*epsilon*4) 
        # we only compute the one half-space of k-vectors, replace 4 with 2. 
        
        """
        Interaction Energy
        """
        charges = self._charges ## array shape N * 1
        r_vec = self._pos ## array shape N * 3
        k_vec = self._get_points_in_sphere() ## array shape n * 3
        k_square = np.sum(k_vec ** 2, 1) ## array shape n * 1
        k_square = np.tile(k_square, 4)
        k_vec_x = np.asarray(np.matmul(k_vec, np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        k_vec_y = np.asarray(np.matmul(k_vec, np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])))
        k_vec_xy = np.asarray(np.matmul(k_vec, np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])))

        k_vec = np.vstack((k_vec, k_vec_x, k_vec_y, k_vec_xy)) ## array shape 4n * 3
        
        krs = np.matmul(k_vec, np.transpose(r_vec)) ## array shape 4n * N

        sk_real = np.matmul(np.cos(krs), charges) ## array shape 4n * 1
        sk_img  = np.matmul(np.sin(krs), charges) ## array shape 4n * 1
        sk2 = sk_real ** 2 + sk_img ** 2  # arrayshape 4n * 1
        
        energy_l = prefac * np.nansum(sk2 * np.exp(-sigma ** 2 * k_square /2)/ k_square) 
        """
        Interaction Forces
        """
        #https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
        expsigma =  np.exp(-sigma ** 2 * k_square /2)/ k_square
        expsigma[expsigma == -inf] = 0
        expsigma[expsigma ==  inf] = 0
        f_real = np.matmul(np.transpose(k_vec), (np.cos(krs) * sk_real[:,None] - np.sin(krs) * sk_img[:,None]) * \
                expsigma[:,None]) ## array shape 3*N 
        
        f_img = np.matmul(np.transpose(k_vec), (np.cos(krs) * sk_img[:,None] + np.sin(krs) * sk_real[:,None]) * \
                expsigma[:,None]) ## array shape 3*N

        #charges = charges.reshape(24,1)
        #print (charges.shape)
        f_real = charges * np.transpose(f_real) * prefac * 4
        f_img = charges * np.transpose(f_img) * prefac * 4

        return energy_l, f_real, f_img

if __name__=="__main__":
    intmolecdist = 0.31
    hoh_angle = 103
    oh_len = 0.97
    box_len = 5
    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    pos = lattice_object() #position of atoms in the primary box
    num_mols = int(pos.shape[0]/3) #num of molecules in one single box 
    qO = -0.82 #charge of oxygen atom
    qH =  0.41 #charge of hydrogen atom
    charges = np.tile(np.array([qO,qH,qH]),num_mols) 
    ewald_test = EwaldSummation(pos, charges, box_len, r_cut=5)
    ewald_test._calc_longrange()

        
        
    
    
     
    
    
    
   
   
   
        
    



    


    

    


    