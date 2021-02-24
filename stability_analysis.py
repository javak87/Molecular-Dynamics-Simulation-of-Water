import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cell_linked_list import CellLinked
from lennard_jones import LennardJones
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity
from vibration_effect import InterMolecularForce
from long_range_initer import CoulombInteraction
from Integrator import Integrator
from FileOperation import FileOperation as FileIO
import h5py
import sys
from apply_periodic_boundary import PeriodicBoundary
from Animation import Animation
import pickle
import time
import math
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

    # calculate angular momentum
    angular_momentum = np.cross(position_array, momentum_array)
    sum_angular_momentum = np.sum (angular_momentum, axis =1)

    # plot momentum in x, y, and z direction
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("Symplectic Analysis of Velocity Verlet Integrator", fontsize=20)

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
    axes[1, 0].set_ylabel('Angular Momentum (mole.Angstroms^2/femtosecond)')
    axes[1, 0].set_ylim([-2e-7, 2e-7])

    axes[1, 1].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_angular_momentum[0:-2,1], color="firebrick")
    #axes[1, 1].set_title('Stability Analysis in the y direction', fontsize=12)
    axes[1, 1].set_xlabel('Time steps')
    axes[1, 1].set_ylabel('Angular Momentum (mole.Angstroms^2/femtosecond)')
    axes[1, 1].set_ylim([-2e-7, 9e-7])
    
    axes[1, 2].plot(np.linspace(1, timesteps, num=timesteps)[0:-1], sum_angular_momentum[0:-2,1], color="firebrick")
    #axes[1, 2].set_title('Stability Analysis in the z direction', fontsize=12)
    axes[1, 2].set_xlabel('Time steps')
    axes[1, 2].set_ylabel('Angular Momentum (mole.Angstroms^2/femtosecond)')
    axes[1, 2].set_ylim([-1e-7, 1e-7])
    
    plt.legend(loc='best')
    plt.show()
    return momentum_array, angular_momentum


def error_analysis () :

    f = open('true_solution.pckl', 'rb')
    postate_true_solution = pickle.load(f)
    f.close()

    f = open('solution_n_3500.pckl', 'rb')
    postate_solution_n_3500 = pickle.load(f)
    f.close()

    f = open('solution_n_3000.pckl', 'rb')
    postate_solution_n_3000 = pickle.load(f)
    f.close()

    f = open('solution_n_2048.pckl', 'rb')
    postate_solution_n_2048 = pickle.load(f)
    f.close()

    f = open('solution_n_1500.pckl', 'rb')
    postate_solution_n_1500 = pickle.load(f)
    f.close()

    f = open('solution_n_1024.pckl', 'rb')
    postate_solution_n_1024 = pickle.load(f)
    f.close()

    f = open('solution_n_800.pckl', 'rb')
    postate_solution_n_800 = pickle.load(f)
    f.close()

    f = open('solution_n_512.pckl', 'rb')
    postate_solution_n_512 = pickle.load(f)
    f.close()

    f = open('solution_n_256.pckl', 'rb')
    postate_solution_n_256 = pickle.load(f)
    f.close()

    f = open('solution_n_200.pckl', 'rb')
    postate_solution_n_170 = pickle.load(f)
    f.close()

    error_n_3500 = np.linalg.norm(postate_true_solution-postate_solution_n_3500)
    error_n_3000 = np.linalg.norm(postate_true_solution-postate_solution_n_3000)

    error_n_2048 = np.linalg.norm(postate_true_solution-postate_solution_n_2048)
    error_n_1500 = np.linalg.norm(postate_true_solution-postate_solution_n_1500)

    error_n_1024 = np.linalg.norm(postate_true_solution-postate_solution_n_1024)
    error_n_800 = np.linalg.norm(postate_true_solution-postate_solution_n_800)

    error_n_512 = np.linalg.norm(postate_true_solution-postate_solution_n_512)
    error_n_256 = np.linalg.norm(postate_true_solution-postate_solution_n_256)
    error_n_170 = np.linalg.norm(postate_true_solution-postate_solution_n_170)

    error = np.array([error_n_3500, error_n_3000, error_n_2048, error_n_1500, error_n_1024, error_n_800, error_n_512, error_n_256, error_n_170])
    deltaT = np.array ([1/3500, 1/3000, 1/2048, 1/1500, 1/1024, 1/800, 1/512, 1/256, 1/200])

    # plot Convergence Analysis
    fig, ax = plt.subplots()
    fig.suptitle("Consistency Analysis", fontsize=20)

    ax.set_xlabel('log (delta T)', fontsize=15)
    ax.set_ylabel('log (Error)', fontsize=15)

    ax.scatter(np.log (error), np.log(deltaT), marker="o", s=100, color='blue', label='Log-Log')
    #ax.scatter(np.log (deltaT), 0.5*np.log(deltaT), marker="o", s=100, color='red', label='semi')
    #ax.scatter(np.log(deltaT), 0.5*np.log(deltaT), marker="o", s=100, color='blue')
    #ax.plot(np.log(deltaT), 2*np.log(deltaT), 'r--')
    #ax.set_xlim([-1e-6, 5e-4])

    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.show()


    return error, deltaT


momentum_array, angular_momentum = plot_momentum (hdf5_file_name, timesteps, save_data_itr, O_mass, H_mass, grid)
error, deltaT = error_analysis ()
