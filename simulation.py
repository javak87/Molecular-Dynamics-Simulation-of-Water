import numpy as np
from cell_linked_list import CellLinked
from lennard_jones import LennardJones
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity
from vibration_effect import InterMolecularForce
from Integrator import Integrator


class Simulation :

        def __init__ (self, grid: np.ndarray,
                        intmolecdist: float, hoh_angle: float,
                        oh_len: float, box_len: float,
                        O_mass: float, H_mass : float,
                        Kb: float, temp: float, sigma: float,
                        epsilon: float, r_cut: float, compmethod: str, k_b: float, tet_eq: float, k_tet: float) :
                
                self.timespan = timespan
                self.intmolecdist = intmolecdist
                self.hoh_angle = hoh_angle
                self.oh_len = oh_len
                self.box_len = box_len
                self.O_mass = O_mass
                self.H_mass = H_mass
                self.Kb = Kb
                self.temp = temp
                self.sigma = sigma
                self.epsilon = epsilon
                self.r_cut = r_cut
                self.compmethod = compmethod
                self.tet_eq = tet_eq
                self.k_tet = k_tet
                self.k_b = k_b
                


        def __call__(self,  timespan)->np.ndarray:

                # Apply position and velocity initialization

                # Initialize position using lattice configuration        
                lattice_object = LatticeConfig (self.intmolecdist, self.hoh_angle, self.oh_len, self.box_len)
                initial_position = lattice_object()

                # Initialize velocity using based on the Boltzmann constant
                vel_object = InitialVelocity (self.O_mass, self.H_mass, self.Kb, self.temp)
                initial_velocity = vel_object (initial_position.shape[0])

                new_postate = initial_position
                new_velocity =initial_velocity

                # create LJ force object
                lj_object = LennardJones(self.sigma, self.epsilon, self.compmethod, self.r_cut, self.box_len)

                # create spring force object
                sp_object = InterMolecularForce (self.oh_len, self.k_b, self.tet_eq, self.k_tet)

                # create Integrator object
                integrator_object = Integrator (O_mass, H_mass)

                for i in range (grid.shape[0]-1) :

                        timespan = (grid [i], grid [i+1])                       

                        lj_force =lj_object (new_postate)

                        sp_force = sp_object(new_postate)

                        force = lj_force + sp_force

                        new_postate[i+1], new_velocity[i+1] = integrator_object (new_postate, new_velocity, force , lj_object, sp_object, timespan)

                return new_postate, new_velocity


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
        timespan = [0, 100]
        grid = np.linspace (timespan[0], timespan[1], 100)


        
        sim_object = Simulation (timespan,
                        intmolecdist, hoh_angle,
                        oh_len, box_len,
                        O_mass, H_mass,
                        Kb, temp)

        new_postate, new_velocity = sim_object (grid)