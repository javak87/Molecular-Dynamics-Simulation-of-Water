import math
import numpy as np

class LatticeConfig :

    def __init__ (self,intmolecdist: float, hoh_angle: float, oh_len: float, box_len: int) :

        self.intmolecdist = intmolecdist
        self.hoh_angle = hoh_angle
        self.oh_len = oh_len
        self.box_len = box_len

        """
        Initialize parameters for lattice configuration

            Parameters:
                intmolecdist (float) : distance between two water molecules (nanometer)
                hoh_angle (float) : degree between two O-H bond in water molecule (degree)
                oh_len (float) : O-H bond length in water molecule (nanometer)
                box_len (float) : the simulation cube length (nanometer)
        """

    def __call__ (self) :
        '''
        creat a lattice configuration for water molecules in a cube
        '''

        return self.config()



    def config (self) :
        '''
        In this function, the number of
        '''
        # compute distance between Oxygene and hydrogene from the top view 
        distance_top_view = self.oh_len*math.sin(0.5*self.hoh_angle*math.pi/180)

        # compute distance between Oxygene and hydrogene from the side view 
        distance_side_view = self.oh_len*math.cos(0.5*self.hoh_angle*math.pi/180)

        # compute distance between two Oxygenes in different layer from
        distance_layers = self.intmolecdist

        # initialize oxygene and hydrogene position as a lattice configuration
        lattice_postate = np.zeros((1, 3))
        lower_bond = 2
        upper_bond = math.floor (self.box_len)
        step_size = math.ceil (2*distance_top_view + self.intmolecdist)
        for i in range (lower_bond, upper_bond, step_size) :
            for j in range (lower_bond, upper_bond, step_size) :
                for k in range (lower_bond, upper_bond, step_size) :
                    oxygene_position = np.array([i, j, k])
                    first_hydroge_position = np.array([i-distance_top_view, j+distance_side_view, k])
                    second_hydroge_position = np.array([i+distance_top_view, j+distance_side_view, k])
                    pos = np.vstack((oxygene_position, first_hydroge_position))
                    pos = np.vstack((pos, second_hydroge_position))
                    lattice_postate = np.vstack((lattice_postate, pos))
       
        return lattice_postate[1:]

if __name__=="__main__":


    intmolecdist = 0.31
    hoh_angle = 103
    oh_len = 0.97
    box_len = 30

    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    lattice_postate = lattice_object()
    print (lattice_postate[3])
