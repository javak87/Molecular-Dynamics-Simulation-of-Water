import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
import numpy as np
from FileOperation import FileOperation as FileIO
from Animation import Animation
from cell_linked_list import CellLinked
from lennard_jones import LennardJones
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity
from vibration_effect import InterMolecularForce
from ewald_summation import EwaldSummation
from Integrator import Integrator
from FileOperation import FileOperation
import h5py
import sys
from apply_periodic_boundary import PeriodicBoundary
import pickle

from visualization import Visualization
from simulation import Simulation
from simulation_visualization_inputs import *


print ('------------\n')
print ('The data for molecular simulation of water stored in the simulation_visualization_inputs.py \n')
print ('------------\n')
print ('You want to visualize the result stored in the hdf5 file or want to simulate the water molecules and after that see the visualization. \n')
print ('if you want to simulate water molecules, this process might the time-consuming. \n')
print ('------------\n')

#vis_or_sim = input ('Please insert "vis" for visualization and insert "sim" for simulation : \n')
vis_or_sim = 'sim'

# Visualize the result stored in the hdf5 file
if vis_or_sim == 'vis' :

    viusl_obj = Visualization (timesteps, update_interval_animation, save_data_itr, hdf5_file_name, x_upper, y_upper, z_upper)
    visual_result = viusl_obj()

# Simulte and the visulize the result

else :
    sim =Simulation (grid, intmolecdist, hoh_angle,
                    oh_len, box_len,
                    O_mass, H_mass,
                    Kb, temp, sigma,
                    epsilon, r_cut, compmethod , k_b, tet_eq, k_tet, save_data_itr,
                    O_charge, H_charge,
                    epszero, sd_dev,
                    k_cut, acc_p)
    postate = sim ()

# visulize the result

    viusl_obj = Visualization (timesteps, update_interval_animation, save_data_itr, hdf5_file_name, x_upper, y_upper, z_upper)
    visual_result = viusl_obj()

# save the last step of molecule's position
f = open('solution_n_200.pckl', 'wb')
pickle.dump(postate, f)
f.close()

print ('The package runs successfully\n')


