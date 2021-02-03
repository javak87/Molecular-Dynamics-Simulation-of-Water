import numpy as np
from lattice_config import LatticeConfig
import math
import transformations as tran

class InterMolecularForce :

    def __init__ (self, r_eq: float, k_b: float, tet_eq: float, k_tet: float) :

        self.r_eq = r_eq
        self.k_b = k_b
        self.tet_eq = tet_eq
        self.k_tet = k_tet

        '''
        this class Calculate stretch force between 2 bonded atoms (O-H).
        and angle bend force between 3 bonded atoms (H-O-H).

        Parameters:
                r_eq (float): Equilibrium bond length [Angstrom] of bond O-H.
                k_b (float): stiness of the covalent bonds [kcal/(mol*A^2)] of bond O-H.
                tet_eq (float) : Equilibrium angle [radi] of bond O-H.
                k_tet (float) : stiness of the angle between both covalent bonds
        '''

    def __call__ (self, postate: np.ndarray) :

        '''
        Parameters:
             postate (np.ndarray) : position state of all water atoms (O and H)
             the arrangment of postate state starts with O position (refer to the first atom)
             and then continue with two H atoms.
             this configuration goes to the end of position state.

        '''

        return self.angle_bend_force (postate) + self.bond_stretch_force(postate)
        


    def bond_stretch_force (self, postate: np.ndarray) :

        '''
        This function calculate Restoring Force using Hooke's law.
        Each oxyegene has been surranded by two Hydrogene atom.
        Each O-H acts as a spring which can creat a force.
        summation of spring force for each O-H impose a force to Oxyegene
        '''
        # creat ghost cell for vectorizing first and the second O-H bond
        gohst_OH1 = np.zeros ((postate.shape[0]+1, postate.shape[1]), dtype=float)


        gohst_OH2 = np.zeros ((postate.shape[0]+2, postate.shape[1]), dtype=float)

        # Inintializing gohst_OH1 and gohst_OH2 with position state
        gohst_OH1[:-1, :] = postate 
        gohst_OH1 = gohst_OH1[1:, :]

        gohst_OH2[:-2, :] = postate
        gohst_OH2 = gohst_OH2[2:, :]

        # Creat vectors for each O-H bond
        vector_OH1 = gohst_OH1 - postate
        vector_OH2 = gohst_OH2 - postate
     

        # Extract vectors of O-H bond
        vector_OH1 = vector_OH1[::3]
        vector_OH2 = vector_OH2[::3]

        # calculate unit vector for both O-H bond
        vector_OH1_hat = vector_OH1/ np.linalg.norm(vector_OH1, axis=1).reshape(-1,1)
        vector_OH2_hat = vector_OH2 / np.linalg.norm(vector_OH2, axis=1).reshape(-1,1)
       

        # calculate Eq. position in 3-D based on the unit vector and Equilibrium bond length
        #eq_OH1 = r_eq*vector_OH1_hat
        #eq_OH2 = r_eq*vector_OH2_hat

        dist_OH1 = np.linalg.norm(vector_OH1)
        dist_OH2 = np.linalg.norm(vector_OH2)

        # Calculate first O-H bond force
        #spring_force_H1 = -k_b* (vector_OH1-eq_OH1)
        spring_force_H1 = -self.k_b* (dist_OH1-self.r_eq)*vector_OH1_hat

        # Calculate second O-H bond force
        spring_force_H2 = -self.k_b* (dist_OH2-self.r_eq)*vector_OH2_hat


        # Calculate imposed force to Oxyegene
        spring_force_O = -1*(spring_force_H1 + spring_force_H2)

        # put forces in a one numpy array
        all_spring_force = np.zeros ((postate.shape[0], postate.shape[1]), dtype=float)
        all_spring_force[0::3] = spring_force_O
        all_spring_force[1::3] = spring_force_H1
        all_spring_force[2::3] = spring_force_H2

        return all_spring_force

    def angle_bend_force (self, postate: np.ndarray) :

        '''
        This function calculate Restoring Force using Hooke's law.
        Each oxyegene has been surranded by two Hydrogene atom.
        H-H bond acts as a spring which can creat a force.    
        '''      

        # extract first hydrogen
        first_H = postate[1::3]

        # extract second hydrogen
        second_H = postate[2::3]

        # extract oxygen 
        oxygen = postate[0::3]   

        # create H1O vector
        vector_OH1 = first_H - oxygen 

        # create H2O
        vector_OH2 = second_H - oxygen

        vector_H2O = oxygen - second_H

        tet_eq_radi = self.tet_eq*math.pi/180

        # the normalized vector in the plane H1OH2 orthogonal to OH1
        pH1 = np.cross(vector_OH1, np.cross(vector_OH1, vector_OH2))

        # the normalized vector in the plane H1OH2 orthogonal to H2O
        pH2 = np.cross(vector_H2O, np.cross(vector_OH1, vector_OH2))

        # calculate the angle between two O-H bond
        vector_OH1_hat = vector_OH1/ np.linalg.norm(vector_OH1, axis=1).reshape(-1,1)
        vector_OH2_hat = vector_OH2 / np.linalg.norm(vector_OH2, axis=1).reshape(-1,1)
        dot_product = np.expand_dims (np.sum(vector_OH1_hat*vector_OH2_hat, axis=1), axis =1)
        angle = np.arccos(dot_product)

        # compute forces
        bend_force_H1 = -self.k_tet*pH1*(angle - tet_eq_radi)/(np.linalg.norm(vector_OH1, axis=1).reshape(-1,1))

        bend_force_H2 = -self.k_tet*pH2*(angle - tet_eq_radi)/(np.linalg.norm(vector_OH2, axis=1).reshape(-1,1))

        bend_force_O = -1*(bend_force_H1+bend_force_H2)

        # combine all force
        all_angle_bend_force = np.zeros ((postate.shape[0], postate.shape[1]), dtype=float)
        all_angle_bend_force[0::3] = bend_force_O
        all_angle_bend_force[1::3] = bend_force_H1
        all_angle_bend_force[2::3] = bend_force_H2

        return all_angle_bend_force
        



if __name__=="__main__":

    intmolecdist = 0.31
    hoh_angle = 103
    oh_len = 0.97
    box_len = 3

    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len=1.97, box_len=10)
    lattice_postate = lattice_object()

    force_object = InterMolecularForce(r_eq=oh_len, k_b=3.5, tet_eq=52, k_tet=1.2)
    spring_force = force_object (lattice_postate)
    
    print (spring_force.sum(0))






        





        


        




