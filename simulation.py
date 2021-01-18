import numpy as np
from cell_linked_list import CellLinked
from lennard_jones import LennardJones
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity
from vibration_effect import InterMolecularForce
from Integrator import Integrator
from FileOperation import FileOperation
import h5py
import sys


class Simulation :

        def __init__ (self, grid: np.ndarray,
                        intmolecdist: float, hoh_angle: float,
                        oh_len: float, box_len: float,
                        O_mass: float, H_mass : float,
                        Kb: float, temp: float, sigma: float,
                        epsilon: float, r_cut: float, compmethod: str, k_b: float, tet_eq: float, k_tet: float, save_data_itr:int) :
                
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
                self.save_data_itr = save_data_itr
                


        def __call__(self)->np.ndarray:

                # Apply position and velocity initialization

                # Initialize position using lattice configuration        
                lattice_object = LatticeConfig (self.intmolecdist, self.hoh_angle, self.oh_len, self.box_len)
                initial_position = lattice_object()

                # Initialize velocity using based on the Boltzmann constant
                vel_object = InitialVelocity (self.O_mass, self.H_mass, self.Kb, self.temp)
                initial_velocity = vel_object (initial_position.shape[0])

                new_postate = initial_position
                new_velocity =initial_velocity

                hdf5_file = h5py.File('data.hdf5','w')

                # save the intial condition
                new_group = hdf5_file.create_group('Timestep Identifier Number = {0}'.format(0))
                new_group.create_dataset('Positions', data=new_postate)
                new_group.create_dataset('Velocities', data=new_velocity)

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

                        new_postate, new_velocity = integrator_object (new_postate, new_velocity, force , lj_object, sp_object, timespan)

                        if i % self.save_data_itr == 0 and i != 0 :

                                new_group = hdf5_file.create_group('Timestep Identifier Number = {0}'.format(i / grid.shape[0]))
                                new_group.create_dataset('Positions', data=new_postate)
                                new_group.create_dataset('Velocities', data=new_velocity)
                       

                        # save new_postate, new_velocity
                        #..
                hdf5_file.close()
                FileOperation.write_hdf5_txt('data.hdf5')
                return new_postate.shape[0]/3


if __name__=="__main__":

        sigma = 3.166 # Angstroms
        epsilon = 0.156 # Kcal/mole
        box_len=1000 # Angstroms
        r_cut= 300 # Angstroms
        intmolecdist = 250 # Angstroms 250 for water
        hoh_angle = 103 # degree
        oh_len = 0.97  # Angstroms
        timespan= (0, 10)
        H_mass = 1.00794
        O_mass = 16
        no_atoms = 6
        Kb = 0.001985875
        temp = 1000

        k_b=3.5
        tet_eq=52
        k_tet=1.2

        save_data_itr = 5


        
        grid = np.linspace (timespan[0], timespan[1], 1000)


        sim =Simulation (grid, intmolecdist, hoh_angle,
                        oh_len, box_len,
                        O_mass, H_mass,
                        Kb, temp, sigma,
                        epsilon, r_cut, 'Cellink_PBC', k_b, tet_eq, k_tet, save_data_itr)
        molecule_count = sim ()
        
        

