import numpy as np
from lennard_jones import LennardJones
from numpy.linalg import inv

class Integrator :

    def __init__ (self, posate: np.ndarray, velocity: np.ndarray, force: np.ndarray, timespan: tuple, mass: np.ndarray) :

        '''
        Parameters
        ----------
        postate: np.ndarray
                 Positions for each particle
        velocity: np.ndarray
                 Velocities for each particle
        force: np.ndarray
                 Forces for each particle
        timespan: tuple
                 Step time indicator
        mass: np.ndarray
                 Particle mass
        '''

        self.posate = posate
        self.velocity = velocity
        self.force = force
        self. timespan =  timespan
        self.mass = mass

    def __call__ (self, box_len: float, sigma: float, epsilon: float, compmethod: str, r_cut: float) :
        
        return self.velocityverlet(box_len, sigma, epsilon, compmethod, r_cut)


    def velocityverlet (self, box_len: float, sigma: float, epsilon: float, compmethod: str, r_cut: float) :
        
        # calculate half step momenta
        momenta_half_step = self.mass * self.velocity + (self.force * (self.timespan[1] - self.timespan[0]) / 2)
        position_full_step = self.posate + (self.timespan[1] - self.timespan[0]) * ( inv(self.mass) ) * momenta_half_step

        # calculate Lennard jones force
        force_object = LennardJones(position_full_step, box_len)
        lj_force = force_object (sigma, epsilon, compmethod,r_cut)

        momenta_full_step = momenta_half_step + ( self.timespan[1] - self.timespan[0] ) * lj_force / 2

        # calculate velocity from momenta
        velocity_full_step = inv(self.mass)*momenta_full_step

        return position_full_step, velocity_full_step