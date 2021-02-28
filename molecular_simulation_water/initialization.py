import numpy as np
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity

class Initializing () :

    def __init__ (self,intmolecdist: float, hoh_angle: float,
                oh_len: float, box_len: int, O_mass: float,
                H_mass: float, Kb: float, temp: float, no_atoms: int) :

        self.intmolecdist = intmolecdist
        self.hoh_angle = hoh_angle
        self.oh_len = oh_len
        self.box_len = box_len
        self.O_mass = O_mass
        self.H_mass = H_mass
        self.Kb = Kb
        self.temp = temp
        self.no_atoms = no_atoms

        """
        Initialize parameters for lattice configuration

            Parameters:
                intmolecdist (float) : distance between two water molecules (Angstroms)
                hoh_angle (float) : degree between two O-H bond in water molecule (degree)
                oh_len (float) : O-H bond length in water molecule (Angstroms)
                box_len (float) : the simulation cube length (Angstroms)
                box_len (float) : simulation cube length (Angstroms)
                O_mass (float) : mass of oxygen atom (grams/mole)
                H_mass (float) : mass of hydrogen atom (grams/mole)
                Kb (float) : Boltzmann constant (kcal/(mol⋅K)) = 0.001985875
                temp (float) : Temperature in the simulation box (Kelvien). room temperature : 298.15
                no_atoms (int) : the number of whole atoms

        """
    def __call__ (self) :

        '''
        this class initializing position and velocity of hydrogen and oxyegen atoms
        Return:
            initial position of all atoms
            initial velocity of all atoms
        '''
        return self.initializingPos(), self.initializingvel ()


    def initializingPos (self) :

        # Initialize position using lattice configuration        
        lattice_object = LatticeConfig (self.intmolecdist, self.hoh_angle, self.oh_len, self.box_len)
        initial_position = lattice_object()

        return initial_position

    def initializingvel (self) :
        # Initialize velocity using based on the Boltzmann constant
        vel_object = InitialVelocity (self.O_mass, self.H_mass, self.Kb, self.temp)
        all_velocities = vel_object (self.no_atoms)

        return all_velocities



