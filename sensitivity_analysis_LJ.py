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
from apply_periodic_boundary import PeriodicBoundary
from simulation import Simulation
from Animation import Animation
import pickle
import time

start1 = time.time()

sigma = 3.166 # Angstroms
epsilon = 0.156 # Kcal/mole
box_len=15 # Angstroms
r_cut= 4 # Angstroms
intmolecdist = 3 # Angstroms 3 for water
hoh_angle = 103 # degree
oh_len = 0.97  # Angstroms
timespan= (0, 1)
H_mass = 1.00794
O_mass = 16
Kb = 0.001985875
temp = 273

k_b=3.5
tet_eq=52
k_tet=1.2

save_data_itr = 5

timesteps = 300


grid = np.linspace (timespan[0], timespan[1], timesteps)


sim =Simulation (grid, intmolecdist, hoh_angle,
                oh_len, box_len,
                O_mass, H_mass,
                Kb, temp, sigma,
                epsilon, r_cut, 'Cellink', k_b, tet_eq, k_tet, save_data_itr)
postate = sim ()

#print (postate)
# save the last step of molecule's position
f = open('solution.pckl', 'wb')
pickle.dump(postate, f)
f.close()
# load the last step of molecule's position
f = open('solution.pckl', 'rb')
postate_solution = pickle.load(f)
f.close()

# load the true solution

f = open('true_solution.pckl', 'rb')
postate_true_solution = pickle.load(f)
f.close()

error = np.linalg.norm(postate_true_solution-postate_solution)

end1 = time.time()
print('Consumed time for force calculation is: \n', end1 - start1)

print('error is: \n', error)

