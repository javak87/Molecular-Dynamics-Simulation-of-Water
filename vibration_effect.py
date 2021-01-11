import numpy as np
from lattice_config import LatticeConfig
import math
import transformations as tran

class InterMolecularForce :

    def __init__ (self, postate: np.ndarray) :

        self.postate= postate

        '''
        this class Calculate stretch force between 2 bonded atoms (O-H).
        and angle bend force between 3 bonded atoms (H-O-H).

        Parameters:
             postate (np.ndarray) : position state of all water atoms (O and H)
             the arrangment of postate state starts with O position (refer to the first atom)
             and then continue with two H atoms.
             this configuration goes to the end of position state.
        '''

    def __call__ (self, r_eq: float, k_b: float, tet_eq: float, k_tet: float) :

        '''
        Parameters:
                r_eq (float): Equilibrium bond length [Angstrom] of bond O-H.
                k_b (float): stiness of the covalent bonds [kcal/(mol*A^2)] of bond O-H.
                tet_eq (float) : Equilibrium angle [radi] of bond O-H.
                k_tet (float) : stiness of the angle between both covalent bonds
        '''

        return self.angle_bend_force (tet_eq, k_tet) + self.bond_stretch_force(r_eq, k_b)
        


    def bond_stretch_force (self, r_eq: float, k_b: float) :

        '''
        This function calculate Restoring Force using Hooke's law.
        Each oxyegene has been surranded by two Hydrogene atom.
        Each O-H acts as a spring which can creat a force.
        summation of spring force for each O-H impose a force to Oxyegene
        '''
        # creat ghost cell for vectorizing first and the second O-H bond
        gohst_OH1 = np.zeros ((self.postate.shape[0]+1, self.postate.shape[1]), dtype=float)


        gohst_OH2 = np.zeros ((self.postate.shape[0]+2, self.postate.shape[1]), dtype=float)

        # Inintializing gohst_OH1 and gohst_OH2 with position state
        gohst_OH1[:-1, :] = self.postate 
        gohst_OH1 = gohst_OH1[1:, :]

        gohst_OH2[:-2, :] = self.postate
        gohst_OH2 = gohst_OH2[2:, :]

        # Creat vectors for each O-H bond
        vector_OH1 = gohst_OH1 - self.postate
        vector_OH2 = gohst_OH2 - self.postate
     

        # Extract vectors of O-H bond
        vector_OH1 = vector_OH1[::3]
        vector_OH2 = vector_OH2[::3]

        # calculate unit vector for both O-H bond
        vector_OH1_hat = vector_OH1/ np.linalg.norm(vector_OH1, axis=1).reshape(-1,1)
        vector_OH2_hat = vector_OH2 / np.linalg.norm(vector_OH1, axis=1).reshape(-1,1)
       

        # calculate Eq. position in 3-D based on the unit vector and Equilibrium bond length
        eq_OH1 = r_eq*vector_OH1_hat
        eq_OH2 = r_eq*vector_OH2_hat

        # Calculate first O-H bond force
        spring_force_H1 = -k_b* (vector_OH1-eq_OH1)

        # Calculate second O-H bond force
        spring_force_H2 = -k_b* (vector_OH2-eq_OH2)


        # Calculate imposed force to Oxyegene
        spring_force_O = -1*(spring_force_H1 + spring_force_H2)

        # put forces in a one numpy array
        all_spring_force = np.zeros ((self.postate.shape[0], self.postate.shape[1]), dtype=float)
        all_spring_force[0::3] = spring_force_O
        all_spring_force[1::3] = spring_force_H1
        all_spring_force[2::3] = spring_force_H2

        return all_spring_force

    def angle_bend_force (self, tet_eq: float, k_tet: float) :

        '''
        This function calculate Restoring Force using Hooke's law.
        Each oxyegene has been surranded by two Hydrogene atom.
        H-H bond acts as a spring which can creat a force.    
        '''
        

        # extract first hydrogen
        first_H = self.postate[1::3]

        # extract second hydrogen
        second_H = self.postate[2::3]

        # extract oxygen 
        oxygen = self.postate[0::3]

        # create H1H2 vector
        vector_H1H2 = second_H - first_H

        # create H2H1 vector
        vector_H2H1 = first_H - second_H       

        # create H1O vector
        vector_OH1 = first_H - oxygen 

        # create H2O
        vector_OH2 = second_H - oxygen 

        # calculate rotation matrix in 3-D by angle k_tet around the norm_vector axis
        tet_eq_radi = tet_eq*math.pi/180

        transformed_vector_OH1_hat = np.zeros ((1, 3), dtype=float)
        transformed_vector_OH2_hat = np.zeros ((1, 3), dtype=float)

        for i in range (0, oxygen.shape[0]) :
            rotation_matrix_OH1 = tran.rotation_matrix(tet_eq_radi, tran.vector_product(vector_OH1[i], vector_OH2[i]))
            rotation_matrix_OH2 = tran.rotation_matrix(-tet_eq_radi, tran.vector_product(vector_OH1[i], vector_OH2[i]))

            # rotate the old positions to get the new one
            new_vector_OH1 = np.dot(vector_OH1[i], rotation_matrix_OH1[:3,:3].T)
            new_vector_OH2 = np.dot(vector_OH2[i], rotation_matrix_OH2[:3,:3].T)

            # compute unit vectors
            new_vector_OH1_hat = tran.unit_vector(new_vector_OH1)
            new_vector_OH2_hat = tran.unit_vector(new_vector_OH2)

            transformed_vector_OH1_hat = np.vstack ([transformed_vector_OH1_hat, new_vector_OH1_hat])
            transformed_vector_OH2_hat = np.vstack ([transformed_vector_OH2_hat, new_vector_OH2_hat])

        # compute Eq. O-H bond vectors
        eq_vector_OH1 = transformed_vector_OH1_hat[1:]
        eq_vector_OH2 = transformed_vector_OH2_hat[1:]

        print ('eq is \n', eq_vector_OH1)

        # compute Eq. H1H2 bond vector
        eq_vector_H2H1 = eq_vector_OH1 - eq_vector_OH2
      

        # compute force on H1 and H2
        bend_force_H2 = -k_tet*(vector_H2H1 - eq_vector_H2H1)
        bend_force_H1 = -k_tet*(vector_H1H2 + eq_vector_H2H1)
        all_angle_bend_force = np.zeros ((self.postate.shape[0], self.postate.shape[1]), dtype=float)

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

    force_object = InterMolecularForce(lattice_postate)
    spring_force = force_object (r_eq=oh_len, k_b=2.5, tet_eq=51, k_tet=2)
    print (spring_force)

    A = np.sum(spring_force, axis=0)
    print (A)





        





        


        




