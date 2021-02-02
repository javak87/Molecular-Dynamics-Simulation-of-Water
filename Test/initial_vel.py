import numpy as np
import math

class InitialVelocity :

    def __init__ (self, O_mass: float, H_mass: float, Kb: float, temp: float):

        '''
        Initializing molecular parameters to compute the initial velocity of oxygen and hydrogen
        Parameters :
                O_mass (float) : mass of oxygen atom (grams/mole)
                H_mass (float) : mass of hydrogen atom (grams/mole)
                Kb (float) : Boltzmann constant (kcal/(mol⋅K)) = 0.001985875
                temp (float) : Temperature in the simulation box (Kelvien). room temperature : 298.15
        '''
        self.O_mass = O_mass
        self.H_mass = H_mass
        self.Kb = Kb
        self.temp = temp
    
    def __call__ (self, no_atoms: int) :

        '''
        this class initialize the atom's velocity based on the Boltzmann constant
        Parameters :
                no_atoms (init) : the number of whole atoms
        Return :
                Velocity of oxygen and hydrogen atoms (Angstroms/femtosecond)     
        '''

        return self.initvelo (no_atoms)

    
    def initvelo (self,no_atoms: int) :

        # compute A constant for oxygen
        A_oxygen = math.sqrt (12*self.Kb*self.temp/self.O_mass)

        # Initialize oxygen atoms velocity
        oxygen_vel = A_oxygen*(np.random.random((math.floor(no_atoms/3),3)) - 0.5*np.ones ((math.floor(no_atoms/3),3), dtype=float))

        # compute A constant for hydrogen
        A_hydrogen = math.sqrt (12*self.Kb*self.temp/self.O_mass)

        # Initialize hydrogen atoms velocity
        hydrogen_vel = A_hydrogen*(np.random.random((math.floor(no_atoms*2/3),3)) - 0.5*np.ones ((math.floor(no_atoms*2/3),3), dtype=float))

        # store all velocities in a numpy array

        all_velocities = np.insert(hydrogen_vel, slice (0, no_atoms-1, 2), oxygen_vel, axis=0)

        return all_velocities

if __name__=="__main__":
    
    vel_object = InitialVelocity (O_mass=16, H_mass=1.00794, Kb=0.001985875, temp=298.15)
    all_velocities = vel_object (no_atoms=6)
    print (all_velocities)
    A = np.sum(all_velocities, axis=0)
    print (A)






        

