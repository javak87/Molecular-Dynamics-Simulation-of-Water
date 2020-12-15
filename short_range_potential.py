import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class ShortRange:

    def __init__(self, dist, state):
        self.dist = dist
        self.state= state

    def __call__(self, sigma: float, epsilon: float) :

        
        return self._compute_stages (sigma, epsilon)
  
    def _compute_stages(self, sigma: float, epsilon: float) -> np.ndarray:

        sigma = 3.166
        epsilon = 1.079
        #potential = np.triu (np.array (4*epsilon*((sigma/self.dist)**12-(sigma/self.dist)**6)), 1)
        diff_state = self.state[np.newaxis, :] - self.state[:, np.newaxis]
        force = np.einsum('ijk,ij->ik', diff_state, 24*sigma*(epsilon**6)*(1-2*((sigma/self.dist)**6))/self.dist**8)
        return force

if __name__=="__main__":

    #state= np.linspace(3.5, 8, 12).reshape (4,3)
    state=np.random.randint(5, size=(3, 3))
    dist = distance.cdist(state, state, 'euclidean')
    np.fill_diagonal(dist, 1.)
    model = ShortRange(dist, state)
    result = model (3.166, 78.2)
    print (result)
    #print (dist)
    print (result.shape)
