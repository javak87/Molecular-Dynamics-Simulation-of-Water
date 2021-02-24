import numpy as np
from scipy import special
from generate_images import GenerateImages
from lattice_config import LatticeConfig
from numpy import inf
from get_recip_kvecs import *



class EwaldSummation :

    '''
    Calculates the electrostatic force of a periodic array of water's atom using
    the Ewald technique.
    '''

    def __init__ (self, O_charge: float, H_charge: float, epszero: float, 
                  box_len: float, sd_dev: float, 
                  #k_cut: float, 
                  k_vec_in_xy: np.ndarray, 
                  k_vec: np.ndarray, 
                  k_square_in_xy: np.ndarray, 
                  k_square: np.ndarray):

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
        self.k_vec_in_xy = k_vec_in_xy
        self.k_vec = k_vec
        self.k_square_in_xy = k_square_in_xy
        self.k_square = k_square
        
        self.bounds = np.array ([self.box_len, self.box_len, self.box_len])             
    
    def __call__ (self, postate) :

        '''
        compute ewald summation
        '''
        real_f = self.real_force(postate)
        recip_f = self.recip_force(postate)
        total_f = real_f + recip_f
        return total_f
    
    def real_force (self, postate) :

        '''
        compute real force using periodic images
        '''
        # compute images (one image in each direction)
        images_object = GenerateImages (postate, self.bounds, self.box_len)
        periodic_images = images_object.expand_images()

        #print (periodic_images)


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
        k_vec = self.k_vec
        k_vec_in = self.k_vec_in_xy
        k_square_in = self.k_square_in_xy
        k_square = self.k_square
        """
        Get atoms configuration
        """
        num_mols = int(postate.shape[0]/3)
        charges = np.tile(np.array([self.O_charge,self.H_charge,self.H_charge]),num_mols) 
        r_vec = postate 

        krs_in = np.matmul(k_vec_in, np.transpose(r_vec)) 
        krs = np.matmul(k_vec, np.transpose(r_vec))
        sk_real_in = np.matmul(np.cos(krs_in), charges) 
        sk_img_in  = np.matmul(np.sin(krs_in), charges) 
        sk_real = np.matmul(np.cos(krs), charges) 
        sk_img  = np.matmul(np.sin(krs), charges) 
        expsigma =  np.exp(-sigma ** 2 * k_square /2)/ k_square
        expsigma_in =  np.exp(-sigma ** 2 * k_square_in /2)/ k_square_in
        
        f_real = np.matmul(np.transpose(k_vec), \
        (np.sin(krs) * sk_real[:,None] - np.cos(krs) * sk_img[:,None]) * expsigma[:,None]) 
        f_real_in = np.matmul(np.transpose(k_vec_in), \
        (np.sin(krs_in) * sk_real_in[:,None] - np.cos(krs_in) * sk_img_in[:,None]) * expsigma_in[:,None])

        """
        we should substract one f_real_in from f_real, since we calculate it twice.
        """ 
        recip_f = prefac * charges[:,None] * (np.transpose(f_real) * 2 - np.transpose(f_real_in))

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
    acc_p = 15
    k_cut = 2 * acc_p / box_len
    k_vec_in_xy, k_vec, k_square_in_xy, k_square = get_recip_kvecs(k_cut, box_len)
    force_obj = EwaldSummation (O_charge, H_charge, epszero, box_len, sd_dev, k_vec_in_xy, k_vec, k_square_in_xy, k_square)
    force_ES = force_obj(postate)
    print (force_ES.sum(0))
    