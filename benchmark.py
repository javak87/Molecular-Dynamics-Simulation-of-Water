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
from Integrator import Integrator
from FileOperation import FileOperation
import h5py
import sys
from apply_periodic_boundary import PeriodicBoundary
import pickle

from visualization import Visualization
from simulation import Simulation
from simulation_visualization_inputs import *
import cProfile, pstats
import re
import time


def benchmark():

    ex_time = []
    molecule_number = []
    box = [16,20, 25]
    box_len = box[0]
    r_cut = 2.5*sigma
    

    for i in range(0,3):

        if i == 0:
            compmethod = 'Naive'
        elif i ==1:
            compmethod = 'Cellink_PBC'
        else:
            compmethod = 'Cellink'

        molecule_number = []

        for j in range(0,len(box)):
            box_len = box[j]
            start_time = time.time()

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
                
                #profiler = cProfile.Profile()
                #profiler.enable()
                #sim()
                #profiler.disable()
                #stats = pstats.Stats(profiler).sort_stats('cumtime')
                #stats.print_stats()
                #stats.dump_stats('./profiler-data')
                
            # visulize the result

                # viusl_obj = Visualization (timesteps, update_interval_animation, save_data_itr, hdf5_file_name, x_upper, y_upper, z_upper)
                # visual_result = viusl_obj()

            # save the last step of molecule's position
            f = open('solution_n_800.pckl', 'wb')
            pickle.dump(postate, f)
            f.close()

            end_time = time.time()

            exc_time = end_time-start_time
            print(postate.shape[0])
            molecule_number.append(postate.shape[0])
        
            print(exc_time)
            ex_time.append(exc_time)
            print(ex_time)

        
        box_len = box[0]
        plt.plot(molecule_number,ex_time,label = compmethod)
        ex_time = []
        
    plt.legend()
    plt.title('Computational Method Execution Time Naive')
    plt.xlabel('Number of Molecules')
    plt.ylabel('Execution Time (s)')
    plt.show()
    return plt



if __name__=="__main__":
    benchmark()


    O2_Count = np.array([64, 125, 216, 343])
    consumed_time_naive = np.array ([189, 602, 1625, 3930])
    consumed_time_linked_cell = np.array ([207, 651, 1742, 4115])
    consumed_time_PD_Linked_cell = np.array ([265, 662, 1747, 4103])

    # plot Convergence Analysis
    fig, ax = plt.subplots()
    fig.suptitle("Consistency Analysis", fontsize=20)

    ax.set_xlabel('log (delta T)', fontsize=15)
    ax.set_ylabel('log (Error)', fontsize=15)

    ax.scatter(np.log(deltaT), np.log (error), marker="o", s=100, color='blue')
    ax.plot (np.log(deltaT), np.log (deltaT**2)+2.55, color='red', label='y=2*x + c')
    plt.legend(loc="lower right")
    plt.legend(prop={"size":16})
