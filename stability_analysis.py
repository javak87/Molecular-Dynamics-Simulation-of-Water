import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cell_linked_list import CellLinked
from lennard_jones import LennardJones
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity
from vibration_effect import InterMolecularForce
from Integrator import Integrator
from FileOperation import FileOperation as FileIO
import h5py
import sys
from apply_periodic_boundary import PeriodicBoundary
from Animation import Animation
import pickle
import time
from visualization import Visualization
from simulation import Simulation
from simulation_visualization_inputs import *



def plot_momentum (hdf5_file_name, timesteps, save_data_itr, O_mass, H_mass, grid) :

    hdf5_file_name = "data.hdf5"
    molecule_count = int(FileIO.return_molecule_count(hdf5_file_name))
    
    position_array = np.array(FileIO.extract_data_to_np_array(timesteps, save_data_itr, molecule_count, hdf5_file_name)[0])
    velocity_array = np.array(FileIO.extract_data_to_np_array(timesteps, save_data_itr, molecule_count, hdf5_file_name)[1])
    # Create a proper mass array
    mass = np.zeros ((position_array.shape[1], position_array.shape[2]), dtype=float)
    mass[0::3] = O_mass
    mass[1::3] = H_mass
    mass[2::3] = H_mass
    mass_array = np.tile(mass,(position_array.shape[0], 1,1))

    # calculate momentum
    momentum_array = mass_array * velocity_array
    sum_momentum = np.sum (momentum_array, axis =1)

    print (np.cross(position_array[1], momentum_array[1]))

    # calculate angular momentum
    angular_momentum = np.cross(position_array, momentum_array)
    sum_angular_momentum = np.sum (angular_momentum, axis =1)

    # plot momentum in x, y, and z direction
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("Stability Analysis", fontsize=20)

    # plot momentum

    axes[0, 0].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_momentum[0:-2,0], color="royalblue")
    axes[0, 0].set_title('X direction', fontsize=12)
    axes[0, 0].set_xlabel('Time steps')
    axes[0, 0].set_ylabel('Momentum (mole.Angstroms/femtosecond)')
    axes[0, 0].set_ylim([-1e-7, 1e-7])

    axes[0, 1].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_momentum[0:-2,1], color="royalblue")
    axes[0, 1].set_title('Y direction', fontsize=12)
    axes[0, 1].set_xlabel('Time steps')
    axes[0, 1].set_ylabel('Momentum (mole.Angstroms/femtosecond)')
    axes[0, 1].set_ylim([-1e-7, 1e-7])

    axes[0, 2].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_momentum[0:-2,1], color="royalblue")
    axes[0, 2].set_title('Z direction', fontsize=12)
    axes[0, 2].set_xlabel('Time steps')
    axes[0, 2].set_ylabel('Momentum (mole.Angstroms/femtosecond)')
    axes[0, 2].set_ylim([-1e-7, 1e-7])

    
    # plot Angular Momentum
    axes[1, 0].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_angular_momentum[0:-2,0], color="firebrick")
    #axes[1, 0].set_title('Stability Analysis in the x direction', fontsize=12)
    axes[1, 0].set_xlabel('Time steps')
    axes[1, 0].set_ylabel('Angular Momentum')
    axes[1, 0].set_ylim([-5e-7, 5e-7])

    axes[1, 1].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_angular_momentum[0:-2,1], color="firebrick")
    #axes[1, 1].set_title('Stability Analysis in the y direction', fontsize=12)
    axes[1, 1].set_xlabel('Time steps')
    axes[1, 1].set_ylabel('Angular Momentum')
    axes[1, 1].set_ylim([-5e-7, 5e-7])

    axes[1, 2].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_angular_momentum[0:-2,1], color="firebrick")
    #axes[1, 2].set_title('Stability Analysis in the z direction', fontsize=12)
    axes[1, 2].set_xlabel('Time steps')
    axes[1, 2].set_ylabel('Angular Momentum')
    axes[1, 2].set_ylim([-5e-7, 5e-7]) 
    plt.legend(loc='best')
    plt.show()
    return momentum_array, angular_momentum


def error_analysis () :

    hdf5_file_name = "true_solution_n_1024.hdf5"
    molecule_count = int(FileIO.return_molecule_count(hdf5_file_name))
    
    position_array = np.array(FileIO.extract_data_to_np_array(timesteps, save_data_itr, molecule_count, hdf5_file_name)[0])
    velocity_array = np.array(FileIO.extract_data_to_np_array(timesteps, save_data_itr, molecule_count, hdf5_file_name)[1])


momentum_array, angular_momentum = plot_momentum (hdf5_file_name, timesteps, save_data_itr, O_mass, H_mass, grid)




    

'''
start1 = time.time()

sim =Simulation (grid, intmolecdist, hoh_angle,
                oh_len, box_len,
                O_mass, H_mass,
                Kb, temp, sigma,
                epsilon, r_cut, 'Cellink', k_b, tet_eq, k_tet, save_data_itr)
postate = sim ()
'''

'''
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
'''

