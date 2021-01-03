import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from cell_linked_list import CellLinked, CellLinkedPeriodic, FindParticleDistance
import time

class LennardJones:

    def __init__(self, postate, box_len) :
        
        self.postate= postate
        self.box_len = box_len

        # compute distances between particles pair

        """
        Parameters:
            postate (np.ndarray): Position of the O2 atoms in 3-D
            postate.shape = (number of O2 atoms, 3)
            box_len : simulation box length (L*L*L)
        """

    def __call__(self, sigma: float, epsilon: float, compmethod: str, r_cut: float) -> np.ndarray :

        """
        Compute the Lennard-Jones force for O2 atoms based on the their distances and positions.


        Parameters:
            sigma (float):   effective atom radius
            epsilon (float): well depth at the minimum potential
            compmethod (str): computation method which is going to be used for lennard Jones force
                compmethod strings:
                    'Naive' = calculate all interactions between the particle's pair without any consideration.
                    'Cellink' = calculate interaction which distance between two pairs is less than the cut-off radius.
                    'Cellink_PBC' = calculate interaction which distance between two pairs is less than the cut-off radius with periodic
                    boundary condition
            r_cut (float) : cut of radius
        """

        # Make sure that no unsupported keywords were passed for the current compmethod

        assert compmethod == 'Normal' or compmethod == 'Cellink' or compmethod == 'Cellink_PBC', 'computation method is selected wrongly. computation method should be "Normal" or "Cellink" or Cellink_PBC'


        if compmethod == 'Normal' :
            return self._compute_all_forces (sigma, epsilon, compmethod)
        elif compmethod == 'Cellink' :
            return self._compute_neighbor_forces (sigma, epsilon, compmethod, r_cut)
        else :
            return self._compute_neighbor_forces_PBC (sigma, epsilon, compmethod, r_cut)

    def _compute_all_forces(self, sigma: float, epsilon: float, compmethod: str) -> np.ndarray:

        """
        This function computes all Lennard-Jones force for O2 atoms without using cut-off method
        """
        #potential = np.triu (np.array (4*epsilon*((sigma/self.dists)**12-(sigma/self.dists)**6)), 1)
        # compute distance between each two pairs
        dists = distance.cdist(self.postate, self.postate, 'euclidean')
        np.fill_diagonal(dists, 1)

        # compute force between each two pairs
        diff_postate = self.postate[np.newaxis, :] - self.postate[:, np.newaxis]
        force = np.einsum('ijk,ij->ik', diff_postate, 24*sigma*(epsilon**6)*(1-2*((sigma/dists)**6))/dists**8)
        return force

    def _compute_neighbor_forces (self, sigma: float, epsilon: float, compmethod: str, r_cut:float) -> np.ndarray :

        """
        This function computes Lennard-Jones force for O2 atoms using cut-off method
        """
        # identified neighbor cells and particles inside of these cells
        result = CellLinked (self.box_len,  r_cut, postate=self.postate)
        
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
            force = np.einsum('ijk, ij->jk', diff_postate, 24*sigma*(epsilon**6)*(1-2*((sigma/dists)**6))/dists**8)

            # Append current force to all force array
            all_force = np.vstack((all_force, force))
           
        
        return all_force[1:, :]


    def _compute_neighbor_forces_PBC (self, sigma: float, epsilon: float, compmethod: str, r_cut: float) -> np.ndarray :

        """
        This function computes Lennard-Jones force for O2 atoms using cut-off method and use periodic boundary condition
        """

        # identified neighbor cells and particles inside of these cells
        
        all_force = np.zeros((1,3), float)

        for i in range (0, self.postate.shape[0]) :
           
            # compute particles inside of each cells and other neighbor cells using periodic boundary condition
            
            object_state = CellLinkedPeriodic (self.box_len, r_cut, self.postate, self.postate[i,:])
            effective_state = object_state._find_particles_inside_neighbor_cells()

            # compute euclidean distance between each target particle and neighbor cell particles
            dists = distance.cdist(effective_state, np.array ([self.postate[i,:]]), 'euclidean')

            # Replace zero by one in self-distance casees
            np.place(dists, dists ==0, 1)

            # Compute forces
            diff_postate = self.postate[i,:][np.newaxis, :] - effective_state[:, np.newaxis]              
            force = np.einsum('ijk, ij->jk', diff_postate, 24*sigma*(epsilon**6)*(1-2*((sigma/dists)**6))/dists**8)

            # Append current force to all force array
            all_force = np.vstack((all_force, force))
           
        
        return all_force[1:, :]
        

     


if __name__=="__main__":

    box_len=5
    r_cut= 1
    postate = box_len * np.random.random_sample((2000, 3))

    start1 = time.time()
    model = LennardJones(postate, box_len)
    result = model (3.166, 1.079, 'Cellink_PBC',r_cut)
    end1 = time.time()
    print(end1 - start1)

    A = np.sum(result, axis=0)
    print (A)



