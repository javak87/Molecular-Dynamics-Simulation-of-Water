from typing import Callable, Protocol, Tuple 
import numpy as np
import math
from models import Model, SIRD
# Observation: The Runge Kutta class hierarchy does not know about the Integrator Protocol

class RungeKutta():
    def __init__(self,a: np.ndarray,b: np.ndarray,c: np.ndarray):
        """
            Runge Kutta method generated by a,b,c of a Butcher tableau.
        """
        # check if this is really the matrix of a Butcher tableau (dimensions)
        self.a = a
        self.b = b
        self.c = c
        assert self.a.shape[0]==self.a.shape[0], "wrong size Butcher tableau (matrix a)"
        assert self.a.shape[0]==self.b.shape[0], "wrong size Butcher tableau (matrix b)"
        assert self.a.shape[0]==self.c.shape[0], "wrong size Butcher tableau (matrix c)"

    def __call__(self,
                 rhs: Callable[[float,np.ndarray],np.ndarray], 
                 timespan: Tuple[float,float],
                 state: np.ndarray):
        """ 
            Compute the next state using the Runge Kutta method and the previous state.

            Parameters:
                rhs (Callable):       right hand side of differential equation
                timespan 
                (List[float,float]):  time interval to integrate
                state (np.ndarray):   state at old time i.e. at timespan[0]

            Returns
                state (np.ndarray):   approximate state at new time (state.shape==initial.shape)
        """
        stages = self._compute_stages(rhs,timespan,state)
        incr = np.zeros_like(state) 
        incr = state.astype(np.float64)
        self.b = self.b.astype(np.float64)

        for b,k in zip(self.b, stages):
            incr = b*k 
        return state + (timespan[1] - timespan[0]) * incr

    def _compute_stages(self, rhs, timespan, state) -> np.ndarray:
        """ 
            Compute the s stages and return them as numpy array of shape == (s,n)
        """
        self.k = np.zeros ((self.a.shape[0],state.shape[0]), dtype=float)
        self.a = self.a.astype(np.float64)
        
        for j in range (self.k.shape[0]) :
            inc = np.zeros ((state.shape[0],), dtype=float)
            for l in range (self.k.shape[0]) :
                inc += self.a[j,l]*self.k[l]
            self.k[j] = rhs (timespan[0]+(timespan[1] - timespan[0])*self.c[j], state + inc)
        return self.k       
        #raise NotImplementedError


class ExplicitRK(RungeKutta):
    def __init__(self,a: np.ndarray,b: np.ndarray,c: np.ndarray):
        super().__init__(a,b,c)
        self.a = np.tril(a, k=-1)      
    
    def _compute_stages(self, rhs, timespan, state) -> np.ndarray:

        return super()._compute_stages(rhs, timespan, state)

class ExplicitEuler (ExplicitRK):
    def __init__(self):
        self.b = np.array([1])
        self.a = np.array ([0])
        self.c = np.array([0])

    def _compute_stages(self, rhs, timespan, state) -> np.ndarray:

        self.k = np.array ([rhs (timespan[0], state )])
        return self.k

class Heun (ExplicitRK):
    def __init__(self):
        self.a = np.array([[0, 0],[1, 0]])
        self.b = np.array([0.5, 0.5])
        self.c = np.array([0, 1])
    
    def _compute_stages(self, rhs, timespan, state) -> np.ndarray:

        return super()._compute_stages(rhs, timespan, state)
        
class RK4 (ExplicitRK):
    def __init__(self):
        self.a = np.array([[0, 0, 0, 0],[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        self.b = np.array([0.166, 0.33, 0.33, 0.166])
        self.c = np.array([0, 0.5, 0.5, 1])
        
    def _compute_stages(self, rhs, timespan, state) -> np.ndarray:

        return super()._compute_stages(rhs, timespan, state)
        

class DIRK(RungeKutta):
    """
        Integrator using diagonally implicit Runge Kutta methods (diagonal of a is non zero)
    """
    def __init__(self,a: np.ndarray, b: np.ndarray, c: np.ndarray):
        # check if a is a lower triangular matrix with zeros in the diagonal 
        super().__init__(a,b,c)
        self.a = np.tril(a, k=-1)

    def _compute_stages(self, rhs, timespan, state) -> np.ndarray:

        raise NotImplementedError


if __name__=="__main__":
    a = np.array ([[0, 0, 0, 0],[0.33, 0, 0, 0],[-0.33, 1, 0, 0],[1, -1, 1, 0]])
    b= np.array ([0.125, 0.375, 0.375, 0.125])
    c= np.array ([0, 0.33, 0.66, 1])
    timespan = (2 , 3)
    solver = ExplicitEuler ()
    rhs = SIRD (0.2, 0.1, 0.01)
    state = rhs (0, np.array ([100, 10, 0, 0]))
    state =  solver (rhs, timespan, state)
    print ('The program runs successfully')

