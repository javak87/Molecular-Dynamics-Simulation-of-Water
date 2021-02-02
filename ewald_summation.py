import numpy as np
from scipy import special
from generate_images import GenerateImages
from numpy import inf

class EwaldSummation :

    '''
    Calculates the electrostatic force of a periodic array of water's atom using
    the Ewald technique.
    '''

    def __init__ (self, O_charge: float, H_charge: float, epszero: float, box_len: float, sd_dev: float) :

        """
        Initializes the Ewald sum. parameters
        Parameters:

        """
        self.box_len = box_len
        self.O_charge = O_charge
        self.H_charge = H_charge
        self.epszero = epszero
        self.sd_dev = sd_dev
        self.bounds = np.array ([self.box_len, self.box_len, self.box_len])             
    
    def __call__ (self, postate) :

        '''
        compute ewald summation
        '''
        return self.real_force(postate)

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
        force_vector = charges * periodic_vector * A_coeff * B_coeff * C_coeff
        return force_vector  

    #def lang_range_force (self, postate) :


    #def self_interaction (self, postate) :
if __name__=="__main__":

    O_charge = -0.834
    H_charge = 0.417
    epszero = 0.8987
    box_len = 10
    sd_dev = 2
    postate = np.random.randint(low=1, high=10, size = (3,3))
    print (postate)
    force_obj = EwaldSummation (O_charge, H_charge, epszero, box_len, sd_dev)
    force_ES = force_obj(postate)
    #print (force_ES)
    ss = force_ES.sum(axis=1)
    print (ss.sum (axis=0))

        
