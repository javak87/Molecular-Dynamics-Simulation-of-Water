import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class LennardJones:

    def __init__(self, dists, postate):
        self.dists = dists
        self.postate= postate
        """
        Parameters:
            dists (np.ndarray): This shows the euclidean distances between O2 atoms.
            dists matrix is a Square matrix which the main diagonal is zero.
            dists.shape= (number of O2 atoms, number of O2 atoms)

            postate (np.ndarray): Position of the O2 atoms in 3-D
            postate.shape = (number of O2 atoms, 3)
        """

    def __call__(self, sigma: float, epsilon: float) -> np.ndarray :

        """
        Compute the Lennard-Jones force for O2 atoms based on the their distances and positions.


        Parameters:
            sigma (float):   effective atom radius
            epsilon (float): well depth at the minimum potential
        """
        return self._compute_stages (sigma, epsilon)
  
    def _compute_stages(self, sigma: float, epsilon: float) -> np.ndarray:

        sigma = 3.166    #O2 Sigma
        epsilon = 1.079  #O2 epsilon
        #potential = np.triu (np.array (4*epsilon*((sigma/self.dists)**12-(sigma/self.dists)**6)), 1)
        diff_postate = self.postate[np.newaxis, :] - self.postate[:, np.newaxis]
        force = np.einsum('ijk,ij->ik', diff_postate, 24*sigma*(epsilon**6)*(1-2*((sigma/self.dists)**6))/self.dists**8)
        return force

if __name__=="__main__":

    postate= np.arange (300).reshape(100,3)
    dists = distance.cdist(postate, postate, 'euclidean')
    np.fill_diagonal(dists, 1)
    model = LennardJones(dists, postate)
    result = model (3.166, 1.079)
    print ("the force matrix is : \n ", result)
