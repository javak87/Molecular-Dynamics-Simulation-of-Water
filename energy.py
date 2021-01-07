#from typing import Protocol
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class PotentialEnergy():
    def __init__(self, state):

        self.state = state
        self.dist = distance.cdist(self.state, self.state, 'euclidean')
        

    def __call__(self,  sigma: float, epsilon: float):

        """ Compute potential energy of the current state.
        
            states shows the position of molecules in 3D

            state.shape[0] : the number of water molecules

            state.shape[1] : molecules position in 3D (x, y, z)

            Parameters:
                sigma (float):       effective atom radius
                epsilon (float):     well depth at the minimum potential
                state (np.ndarray):   state at old time i.e. at timespan[0]

            Returns
                state (np.ndarray):   approximate state at new time (state.shape==initial.shape)
        """
        
        "calculate euclidean distance between water in a Square matrix with zero diameter"


                
        return self._compute_stages (self.dist)

    def _compute_stages(self, dist) -> np.ndarray :
        
        return self.dist

class ExternalEnergy (PotentialEnergy):
    def __init__(self):
        super().__init__(self.state)
    
    def _compute_stages(self, dist) -> np.ndarray :
    
        pass
    
class ShortRange (ExternalEnergy) :

    def __init__(self):
        super().__init__(self.state)
    
    def _compute_stages(self, sigma: float, epsilon: float) -> np.ndarray:

        sigma = 3.166
        epsilon = 78.2
        return 4*epsilon*((sigma/self.dist)**12-(sigma/self.dist)**6)
    
class LongRange (ExternalEnergy) :
    def __init__(self):

        pass
    
    def _compute_stages(self, state) -> np.ndarray:
    
        raise NotImplementedError
    
class InternalEnergy (PotentialEnergy):
    def __init__(self):

        pass
    
    def _compute_stages(self, state) -> np.ndarray:
    
        pass

class HarmonicBondVibration (InternalEnergy):

    def __init__(self):

        pass
    
    def _compute_stages(self, state) -> np.ndarray:
    
        raise NotImplementedError

class HarmonicAngleVibration (InternalEnergy):

    def __init__(self):

        pass
    
    def _compute_stages(self, state) -> np.ndarray:
    
        raise NotImplementedError

 

if __name__=="__main__":

    r = np.linspace(3.5, 8, 99).reshape (33,3)

    model = ShortRange(r)
    result = model ( 0.0103, 3.4)
    plt.plot(r, model (r, 0.0103, 3.4))
    plt.xlabel(r'$r$/Å')
    plt.ylabel(r'$f$/eVÅ$^{-1}$')
    plt.show()

