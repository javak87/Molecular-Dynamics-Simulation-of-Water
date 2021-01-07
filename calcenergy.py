from typing import Callable, Protocol, Tuple 
import numpy as np
from energy import Energy, ExternalEnergy, ShortRange, LongRange 
#from models import Model, SIRD

class EnergyType(Protocol):
    def __call__(self, 
                 rhs: Callable[[float,np.ndarray],np.ndarray], 
                 timespan: Tuple[float,float],
                 initial: np.ndarray)->np.ndarray:
        pass

def FlexibleSPC (energy_type: EnergyType,
           rhs: Callable[[float,np.ndarray],np.ndarray],
           grid: np.ndarray,
           initial: np.ndarray) -> np.ndarray: 
   
    """ Approximates the differential equation at all given points in time (grid)

        The differential equation reads

        y'(t) = f(t,y(t))
        y(0)  = y_0

        where f: [0,T] x R^n -> R^n and y_0 \in R^n

        Parameters:
            integrator (Integrator): single step numerical method satisfying the Integrator protocol

            rhs (Callable):          The function f taking current time and state 
                                     and returning y'(t) as numpy array with shape (n,) 

            grid (np.ndarray):       array collecting points in time i.e.
                                     [0,...,t_i,t_{i+1},...,T]

            initial (np.ndarray):    initial state y_0 


        Returns:
            y (np.ndarray):          The states at the time points given in grid i.e.
                                     [y(0),y(t_1),...,y(t_i),...,y(T)]
    """
        short_range= ShortRange().__call__ (rhs, timespan, new_state[i])
        long_range= LongRange().__call__ (rhs, timespan, new_state[i])
        harmonic_bond_vibration= HarmonicBondVibration().__call__ (rhs, timespan, new_state[i])
        harmonic_angle_vibration= HarmonicAngleVibration().__call__ (rhs, timespan, new_state[i])
        total_energy= short_range +long_range + harmonic_bond_vibration + harmonic_angle_vibration
    return total_energy
 
