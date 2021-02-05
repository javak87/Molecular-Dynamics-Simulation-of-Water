import numpy as np
from lennard_jones import LennardJones
from numpy.linalg import inv
from vibration_effect import InterMolecularForce
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity

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
        force = spring_force

        momenta_full_step = momenta_half_step + ( timespan[1] - timespan[0] ) * force / 2

        # calculate velocity from momenta
        velocity_full_step = np.dot (inv(mass_matrix), momenta_full_step)

        return position_full_step, velocity_full_step

if __name__=="__main__":

    
    sigma = 3.166 # Angstroms
    epsilon = 0.156 # Kcal/mole
    box_len=1000 # Angstroms
    r_cut= 500 # Angstroms
    intmolecdist = 250 # Angstroms
    hoh_angle = 103 # degree
    oh_len = 0.97  # Angstroms
    timespan= (0,0.1)
    H_mass = 1.00794
    O_mass = 16
    no_atoms = 6
    Kb = 0.001985875
    temp = 298.15
    compmethod = 'Cellink_PBC'

    k_b=3.5
    tet_eq=52
    k_tet=1.2
    # Apply position and velocity initialization

    # Initialize position using lattice configuration        
    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    initial_position = lattice_object()

    # Initialize velocity using based on the Boltzmann constant
    vel_object = InitialVelocity (O_mass, H_mass, Kb, temp)
    initial_velocity = vel_object (initial_position.shape[0])

    new_postate = initial_position
    new_velocity =initial_velocity

    # create LJ force object
    lj_object = LennardJones(sigma, epsilon, compmethod, r_cut, box_len)

    # create spring force object
    sp_object = InterMolecularForce (oh_len, k_b, tet_eq, k_tet)

    integrator_object = Integrator (O_mass, H_mass)

    timespan = [0, 0.1]                       

    lj_force =lj_object (new_postate)

    sp_force = sp_object(new_postate)

    force = lj_force + sp_force

    new_pos, new_vel = integrator_object (new_postate, new_velocity, force , lj_object, sp_object, timespan)

    print ('new position : \n', new_pos)
