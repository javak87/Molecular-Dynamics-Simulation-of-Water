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
from get_recip_kvecs import *
from simulation import Simulation
from simulation_visualization_inputs import *
import cProfile, pstats
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

compmethod = 'Naive'

def naive_simulation():
    
    compmethod = 'Naive'
    sim =Simulation (grid, intmolecdist, hoh_angle,
                    oh_len, box_len,
                    O_mass, H_mass,
                    Kb, temp, sigma,
                    epsilon, r_cut, compmethod , k_b, tet_eq, k_tet, save_data_itr,
                    O_charge, H_charge,
                    epszero, sd_dev,
                    #k_cut, acc_p,
                    k_vec_in_xy, k_vec, k_square_in_xy, k_square)

    postate = sim()
    return postate

def pbc_simulation():
    
    compmethod = 'Cellink_PBC'
    sim =Simulation (grid, intmolecdist, hoh_angle,
                    oh_len, box_len,
                    O_mass, H_mass,
                    Kb, temp, sigma,
                    epsilon, r_cut, compmethod , k_b, tet_eq, k_tet, save_data_itr,
                    O_charge, H_charge,
                    epszero, sd_dev,
                    #k_cut, acc_p,
                    k_vec_in_xy, k_vec, k_square_in_xy, k_square)

    postate = sim()
    return postate

def cellink_simulation():
    
    compmethod = 'Cellink'
    sim =Simulation (grid, intmolecdist, hoh_angle,
                    oh_len, box_len,
                    O_mass, H_mass,
                    Kb, temp, sigma,
                    epsilon, r_cut, compmethod , k_b, tet_eq, k_tet, save_data_itr,
                    O_charge, H_charge,
                    epszero, sd_dev,
                    #k_cut, acc_p,
                    k_vec_in_xy, k_vec, k_square_in_xy, k_square)

    postate = sim()
    return postate

if __name__== "__main__":
    
    #Call diagram for Naive
    graphviz = GraphvizOutput()
    graphviz.output_file = "naive.png"
   
    profiler = cProfile.Profile()
    profiler.enable()
    with PyCallGraph(output=graphviz):            
        naive_simulation()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('./profiler-naive-data')
    """
    #Call diagram for PBC
    graphviz = GraphvizOutput()
    graphviz.output_file = "pbc.png"
   
    profiler = cProfile.Profile()
    profiler.enable()
    with PyCallGraph(output=graphviz):            
        pbc_simulation()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('./profiler-pbc-data')

    #Call diagram for Cellink
    graphviz = GraphvizOutput()
    graphviz.output_file = "cellink.png"
   
    profiler = cProfile.Profile()
    profiler.enable()
    with PyCallGraph(output=graphviz):            
        cellink_simulation()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('./profiler-cellink-data')
    """