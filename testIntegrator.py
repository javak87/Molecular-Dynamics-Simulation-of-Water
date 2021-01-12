import numpy as np
import math
from Integrator import Integrator
from lattice_config import LatticeConfig
from lennard_jones import LennardJones
from initial_vel import InitialVelocity

def integrator_test():
    
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

    hydrogen_mass = H_mass * np.ones( no_atoms * 2/3 , dtype=float)
    oxygen_mass = O_mass * np.ones( no_atoms / 3 , dtype=float )
    mass = np.insert(hydrogen_mass, slice (0, no_atoms-1, 2), oxygen_mass, axis=0)

    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    postate = lattice_object()

    force_object = LennardJones(postate, box_len)
    lj_force_cell_linked = force_object (sigma, epsilon, 'Cellink',r_cut)
    lj_force_periodic = force_object (sigma, epsilon, 'Cellink_PBC',r_cut)
    lj_force_naive = force_object (sigma, epsilon, 'Naive',r_cut)

    vel_object = InitialVelocity ( O_mass, H_mass, Kb, temp )
    all_velocities = vel_object ( no_atoms)

    print(postate)
    print(all_velocities)

    verlet_cell_linked = Integrator(postate, all_velocities, lj_force_cell_linked, timespan, mass)
    postate, all_velocities = verlet_cell_linked.velocityverlet

    print(postate)
    print(all_velocities)

    verlet_periodic = Integrator(postate, all_velocities, lj_force_periodic, timespan, mass)
    postate, all_velocities = verlet_periodic.velocityverlet

    print(postate)
    print(all_velocities)

    verlet_naive = Integrator(postate, all_velocities, lj_force_naive, timespan, mass)
    postate, all_velocities =  verlet_naive.velocityverlet

    print(postate)
    print(all_velocities)

if __name__ == "__main__":
    integrator_test()
