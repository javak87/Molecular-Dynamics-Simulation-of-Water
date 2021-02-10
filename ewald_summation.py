import numpy as np
from scipy import special
from generate_images import GenerateImages
from lattice_config import LatticeConfig
from numpy import inf
from math import pi, floor


class EwaldSummation :

    '''
    Calculates the electrostatic force of a periodic array of water's atom using
    the Ewald technique.
    '''

    def __init__ (self, O_charge: float, H_charge: float, epszero: float, 
                  box_len: float, sd_dev: float, k_cut: float, acc_p: int) :

        """
        Initializes the Ewald sum. parameters
        Parameters:

        """
        self.box_len = box_len
        self.volume = self.box_len ** 3
        self.O_charge = O_charge
        self.H_charge = H_charge
        self.epszero = epszero
        self.sd_dev = sd_dev
        self.k_cut = k_cut
        self.acc_p = acc_p
        self.bounds = np.array ([self.box_len, self.box_len, self.box_len])             
    
    def __call__ (self, postate) :

        '''
        compute ewald summation
        '''
        real_f = self.real_force(postate)
        recip_f = self.recip_force(postate)
        total_f = real_f + recip_f
        return total_f
    
    def _get_points_in_sphere(self):
        radius = floor(self.k_cut * self.box_len / (2 * pi))
        k_vec = np.empty((0,3), int)
        for i in range(radius+1):
            for j in range(radius+1):
                for k in range(radius+1):
                    if (( i * i + j * j + k * k)<= radius ** 2):
                        k_vec = np.append(k_vec,np.array([[i,j,k]]),axis=0)
        k_vec = k_vec * 2 * pi/self.box_len
        return k_vec

    def real_force (self, postate) :

        '''
        compute real force using periodic images
        '''
        # compute images (one image in each direction)
        images_object = GenerateImages (postate, self.bounds, self.box_len)
        periodic_images = images_object.expand_images()

        print (periodic_images)


        # assign oxygen and Hydrogen charge to all atoms
        point_charge = np.zeros((postate.shape[0],1))
        point_charge[0::3] = self.O_charge
        point_charge[1::3] = self.H_charge
        point_charge[2::3] = self.H_charge
        
        charge_array = np.repeat(point_charge, 27).reshape (periodic_images.shape[0],1)

        charges = point_charge[:, np.newaxis]*charge_array[np.newaxis, :]


        # compute ri-rj-nL vector
        periodic_vector =  postate[:, np.newaxis] - periodic_images[np.newaxis, :]
        # compute vector norm (|ri-rj-nL|)
        norm_periodic_vector = np.linalg.norm (periodic_vector, axis=2).reshape(periodic_vector.shape[0], periodic_vector.shape[1], 1)


        # calculate coefficients for short-range interaction 

        A_coeff = special.erfc(norm_periodic_vector/(np.sqrt(2)*self.sd_dev))/norm_periodic_vector
        A_coeff = np.where (A_coeff == inf, 0, A_coeff)
             
        B_coeff = np.sqrt(2/np.pi)*np.exp(-1*(norm_periodic_vector**2)/(2*self.sd_dev**2))/self.sd_dev

        C_coeff = 1/norm_periodic_vector**2
        C_coeff = np.where (C_coeff == inf, 0, C_coeff)

        # compute short-range force vector
        real_f = (charges * periodic_vector * A_coeff * B_coeff * C_coeff).sum(axis = 1)
        return real_f 
    def recip_force(self, postate) :
        """
        Calculate pre parameters 
        """
        epsilon = self.epszero
        volume = self.volume
        sigma = self.sd_dev
        prefac = 1/(volume*epsilon) 
        # we only compute the one half-space of k-vectors, replace 4 with 2. 
        
        """
        Interaction Energy
        """
        num_mols = int(postate.shape[0]/3)
        charges = np.tile(np.array([self.O_charge,self.H_charge,self.H_charge]),num_mols) 
        ## array shape N * 1    N atoms
        r_vec = postate ## array shape N * 3
        k_vec = self._get_points_in_sphere() ## array shape n * 3  x,y,z positive kvectors
        k_square = np.sum(k_vec ** 2, 1) ## array shape n * 1
        k_square = np.tile(k_square, 4) ## array shape 4n*1 upper sphere 
        k_vec_x = np.asarray(np.matmul(k_vec, np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        k_vec_y = np.asarray(np.matmul(k_vec, np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])))
        k_vec_xy = np.asarray(np.matmul(k_vec, np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])))

        k_vec = np.vstack((k_vec, k_vec_x, k_vec_y, k_vec_xy)) ## array shape 4n * 3  upper sphere kvectors
        
        krs = np.matmul(k_vec, np.transpose(r_vec)) ## array shape 4n * N   dot product of k and r 

        sk_real = np.matmul(np.cos(krs), charges) ## array shape 4n * 1
        # sk_img  = np.matmul(np.sin(krs), charges) ## array shape 4n * 1
        # sk2 = sk_real ** 2 + sk_img ** 2  # arrayshape 4n * 1 structure factor square 
        
        # energy_l = prefac * np.nansum(sk2 * np.exp(-sigma ** 2 * k_square /2)/ k_square) 
        """
        Interaction Forces
        """
        #https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
        expsigma =  np.exp(-sigma ** 2 * k_square /2)/ k_square
        expsigma[expsigma == -inf] = 0
        expsigma[expsigma ==  inf] = 0
        f_real = np.matmul(np.transpose(k_vec), (np.sin(krs) * sk_real[:,None]) * \
                expsigma[:,None]) ## array shape 3*N 
        
        recip_f = charges[:,None] * np.transpose(f_real) * prefac * 2

        return recip_f
    
    #def self_interaction (self, postate) :
if __name__=="__main__":
    intmolecdist = 0.31
    hoh_angle = 103
    oh_len = 0.97
    box_len = 10
    sd_dev = 2
    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    postate = lattice_object()
    O_charge = -0.834
    H_charge = 0.417
    epszero = 0.8987
    k_cut = box_len
    acc_p = 12
    print (postate)
    force_obj = EwaldSummation (O_charge, H_charge, epszero, box_len, sd_dev, k_cut, acc_p)
    force_ES = force_obj(postate)
    print (force_ES)
    

        
