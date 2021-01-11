from typing import Callable, Protocol, Tuple 
import numpy as np
from cell_linked_list import CellLinked
from lennard_jones import LennardJones
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity

class Simulation(Protocol):

    def __call__(self, 
                 timespan: Tuple[float,float])->np.ndarray:
        pass

def FlexibleSPC (self, grid: np.ndarray) -> np.ndarray: 
   
        # Apply position and velocity initialization

        # Initialize position using lattice configuration        
        lattice_object = LatticeConfig (self.intmolecdist, self.hoh_angle, self.oh_len, self.box_len)
        initial_position = lattice_object()

        # Initialize velocity using based on the Boltzmann constant
        vel_object = InitialVelocity (self.O_mass, self.H_mass, self.Kb, self.temp)
        initial_velocities = vel_object (self.no_atoms)

        # compute lennard jones force
        force_object = LennardJones(postate, box_len)
        lj_force = force_object (sigma, epsilon, 'Cellink',r_cut)


 
