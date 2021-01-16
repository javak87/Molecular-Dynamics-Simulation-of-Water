import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from cell_linked_list import CellLinked, CellLinkedPeriodic, FindParticleDistance
import time
from lattice_config import LatticeConfig

class LennardJones:

    def __init__(self, sigma: float, epsilon: float, compmethod: str, r_cut: float, box_len: float):
        
        """
        Compute the Lennard-Jones force for O2 atoms based on the their distances and positions.


        Parameters:
            sigma (float):   effective atom radius (Angstroms)
            epsilon (float): well depth at the minimum potential (Kcal/mole)
            compmethod (str): computation method which is going to be used for lennard Jones force
                compmethod strings:
                    'Naive' = calculate all interactions between the particle's pair without any consideration.
                    'Cellink' = calculate interaction which distance between two pairs is less than the cut-off radius.
                    'Cellink_PBC' = calculate interaction which distance between two pairs is less than the cut-off radius with periodic
                    boundary condition
            r_cut (float) : cut of radius (Angstroms)
            box_len (Angstroms) : simulation box length (L*L*L)

        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.compmethod = compmethod
        self.r_cut = r_cut
        self.box_len =box_len

    def __call__(self, postate) -> np.ndarray :

        """
        Parameters:
            postate (np.ndarray): Position of all atoms in 3-D
            postate.shape = (number of all atoms, 3)
        Return:
            force (np.ndarray) : Lennard jones force
            force.shape = (number of all atoms, 3)
        """
        # extract oxyegene atom from position state of all atoms for computing lennard jones force
        postate= postate[::3]

        # Make sure that no unsupported keywords were passed for the current compmethod

        assert self.compmethod == 'Naive' or self.compmethod == 'Cellink' or self.compmethod == 'Cellink_PBC', 'computation method is selected wrongly. computation method should be "Normal" or "Cellink" or Cellink_PBC'


        if self.compmethod == 'Naive' :
            return self._compute_all_forces (postate)
        elif self.compmethod == 'Cellink' :
            return self._compute_neighbor_forces (postate)
        else :
            return self._compute_neighbor_forces_PBC (postate)

    def _compute_all_forces(self, postate) -> np.ndarray:

        """
        This function computes all Lennard-Jones force (Kcal/mole-Angstrom) for O2 atoms without using cut-off method
        """
        #potential = np.triu (np.array (4*epsilon*((sigma/self.dists)**12-(sigma/self.dists)**6)), 1)
        # compute distance between each two pairs
        dists = distance.cdist(postate, postate, 'euclidean')
        np.fill_diagonal(dists, 1)

        # compute force between each two pairs
        diff_postate = postate[np.newaxis, :] - postate[:, np.newaxis]
        force = np.einsum('ijk,ij->ik', diff_postate, 24*self.sigma*(self.epsilon**6)*(1-2*((self.sigma/dists)**6))/dists**8)
        
        # add zero rows to force numpy array instead of hydrogene atoms.
        # because lennard jones is only valid for oxyegene, lennard jones force for hydrogenes is zero
        first_hydrogene_index = range (1, postate.shape[0]+1, 1)
        second_hydrogene_index = range (1, 2*postate.shape[0], 2)
        add_first_hydro_force = np.insert(force, first_hydrogene_index, np.array((0.0, 0.0, 0.0)), 0)
        all_force = np.insert(add_first_hydro_force, second_hydrogene_index, np.array((0.0, 0.0, 0.0)), 0)
        return all_force

    def _compute_neighbor_forces (self, postate) -> np.ndarray :

        """
        This function computes Lennard-Jones force (Kcal/mole-Angstrom) for O2 atoms using cut-off method
        """
        # identified neighbor cells and particles inside of these cells
        result = CellLinked (self.box_len,  self.r_cut, postate)
        
        # compute head and list array
        head_arr, list_arr = result ()
        neighbor_cells = result._find_neighbor_cells(head_arr)
        
        all_force = np.zeros((1,3), float)

        for i in range (0, postate.shape[0]) :

            # find cell index of a specific particle            
            cellindex = result._cellindex (postate[i,:])

            # find neighbor cell index based on the cell index
            neighbor_cells = result._find_neighbor_index (cellindex)
            
            # compute particles inside of each cells and other neighbor cells
            effective_state = result._find_particles_inside_neighbor_cells (list_arr, head_arr, neighbor_cells)

            state_object = FindParticleDistance (postate[i,:], effective_state, r_cut)
            effective_state = state_object._find_particles_distance_inside_neighbor_cells()          

            # compute euclidean distance between each original particle and other particles
            dists = distance.cdist(effective_state, np.array ([postate[i,:]]), 'euclidean')

            # Replace zero by one in self-distance casees
            np.place(dists, dists ==0, 1)

            # Compute forces
            diff_postate = postate[i,:][np.newaxis, :] - effective_state[:, np.newaxis]              
            force = np.einsum('ijk, ij->jk', diff_postate, 24*self.sigma*(self.epsilon**6)*(1-2*((self.sigma/dists)**6))/dists**8)

            # Append current force to all force array
            all_force = np.vstack((all_force, force))

        force = all_force[1:, :]
        first_hydrogene_index = range (1, force.shape[0]+1, 1)
        second_hydrogene_index = range (1, 2*force.shape[0], 2)
        add_first_hydro_force = np.insert(force, first_hydrogene_index, np.array((0.0, 0.0, 0.0)), 0)
        all_force = np.insert(add_first_hydro_force, second_hydrogene_index, np.array((0.0, 0.0, 0.0)), 0)   
        
        return all_force


    def _compute_neighbor_forces_PBC (self, postate) -> np.ndarray :

        """
        This function computes Lennard-Jones force (Kcal/mole-Angstrom) for O2 atoms using cut-off method and use periodic boundary condition
        """

        # identified neighbor cells and particles inside of these cells
        
        all_force = np.zeros((1,3), float)

        for i in range (0, postate.shape[0]) :
           
            # compute particles inside of each cells and other neighbor cells using periodic boundary condition
            
            object_state = CellLinkedPeriodic (self.box_len, self.r_cut, postate, postate[i,:])
            effective_state = object_state._find_particles_inside_neighbor_cells()

            # compute euclidean distance between each target particle and neighbor cell particles
            dists = distance.cdist(effective_state, np.array ([postate[i,:]]), 'euclidean')

            # Replace zero by one in self-distance casees
            np.place(dists, dists ==0, 1)

            # Compute forces
            diff_postate = postate[i,:][np.newaxis, :] - effective_state[:, np.newaxis]              
            force = np.einsum('ijk, ij->jk', diff_postate, 24*self.sigma*(self.epsilon**6)*(1-2*((self.sigma/dists)**6))/dists**8)

            # Append current force to all force array
            all_force = np.vstack((all_force, force))
           
        force = all_force[1:, :]
        first_hydrogene_index = range (1, force.shape[0]+1, 1)
        second_hydrogene_index = range (1, 2*force.shape[0], 2)
        add_first_hydro_force = np.insert(force, first_hydrogene_index, np.array((0.0, 0.0, 0.0)), 0)
        all_force = np.insert(add_first_hydro_force, second_hydrogene_index, np.array((0.0, 0.0, 0.0)), 0)   

        return all_force
        

     


if __name__=="__main__":

    sigma = 3.166 # Angstroms
    epsilon = 0.156 # Kcal/mole
    box_len=1000 # Angstroms
    r_cut= 500 # Angstroms
    intmolecdist = 250 # Angstroms
    hoh_angle = 103 # degree
    oh_len = 0.97  # Angstroms
    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    postate = lattice_object()


    #postate = box_len * np.random.random_sample((2000, 3))
    

    start1 = time.time()
    model = LennardJones(sigma, epsilon, 'Cellink_PBC', r_cut, box_len)
    result = model (postate)
    end1 = time.time()
    print('Consumed time for periodic cell_linked list force calculation (24000 particles) is: \n', end1 - start1)

    A = np.sum(result, axis=0)
    #print (A)





