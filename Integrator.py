import numpy as np
from lennard_jones import LennardJones
from numpy.linalg import inv
from vibration_effect import InterMolecularForce

class Integrator :

    def __init__ (self, O_mass : float, H_mass: float) :

        '''
        Parameters
        ----------
        postate: np.ndarray
                 Positions for each particle
        velocity: np.ndarray
                 Velocities for each particle
        force: np.ndarray
                 Forces for each particle
        timespan: tuple
                 Step time indicator
        O_mass: float
                oxygen mass
        H_mass: float
                hydrogen mass
                
        '''

        self.O_mass = O_mass
        self.H_mass = H_mass



    def __call__ (self, postate: np.ndarray, velocity: np.ndarray, force, lj_object, sp_object, timespan) :

        
        return self.velocityverlet(postate, velocity, force , lj_object, sp_object, timespan)


    def velocityverlet (self, postate: np.ndarray, velocity: np.ndarray, force, lj_object, sp_object, timespan) :
        
        # create diagonal mass numpy array
        mass_matrix = np.zeros((postate.shape[0], postate.shape[0]), float)
        rept_mat = np.zeros((1, postate.shape[0]), float)
        rept_mat[0, 0::3] = self.O_mass
        rept_mat[0, 1::3] = self.H_mass
        rept_mat[0, 2::3] = self.H_mass
        np.fill_diagonal(mass_matrix, rept_mat)
        mass_matrix = mass_matrix

        # extract diagonal elements of mass matrix
        diag_mass = mass_matrix.diagonal()[..., None]
        
        a = np.hstack((diag_mass,diag_mass))
        diag_mass = np.hstack((diag_mass,a))

        diag_mass = diag_mass
        
        # calculate half step momenta
        momenta_half_step = diag_mass * velocity + (force * (timespan[1] - timespan[0]) / 2)
        position_full_step = postate + (timespan[1] - timespan[0]) * np.dot (inv(mass_matrix), momenta_half_step) 

        # calculate forces
        lj_force = lj_object (position_full_step)
        spring_force = sp_object(position_full_step)
        force = lj_force + spring_force

        momenta_full_step = momenta_half_step + ( timespan[1] - timespan[0] ) * force / 2

        # calculate velocity from momenta
        velocity_full_step = np.dot (inv(mass_matrix), momenta_full_step)

        return position_full_step, velocity_full_step