import numpy as np
from lattice_config import LatticeConfig
from initial_vel import InitialVelocity
from vibration_effect import InterMolecularForce
from Integrator import Integrator
from lennard_jones import LennardJones

sigma = 3.166 # Angstroms
epsilon = 0.156 # Kcal/mole
box_len=1000 # Angstroms
r_cut= 500 # Angstroms
intmolecdist = 250 # Angstroms
hoh_angle = 103 # degree
oh_len = 0.97  # Angstroms
k_b=3.5
tet_eq=52
k_tet=1.2
O_mass=16
H_mass=1.00794
Kb=0.001985875
temp=298.15

lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
postate = lattice_object()

# Initialize velocity using based on the Boltzmann constant
vel_object = InitialVelocity (O_mass, H_mass, Kb, temp)
velocity = vel_object (postate.shape[0])

lj_object = LennardJones(sigma, epsilon, 'Cellink_PBC', r_cut, box_len)
lj_force =lj_object (postate)

sp_object = InterMolecularForce (oh_len, k_b, tet_eq, k_tet)
sp_force = sp_object(postate)

force = lj_force + sp_force

timespan = [2, 3]

integrator_object = Integrator (O_mass, H_mass)

new_postate, new_velocity= integrator_object (postate, velocity, force , lj_object, sp_object, timespan)

print ('old is : \n', postate)
print ('new is : \n', new_postate)






