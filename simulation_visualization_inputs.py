import numpy as np
'''
These are the input for molecular dynamic of water.
'''

'''
Simulation Parameters :
'''
# position Initialization parameters
box_len=10 # Angstroms
intmolecdist = 4 # Angstroms 1.5 for water

# velocity Initialization parameters
H_mass = 1.00794
O_mass = 16
Kb = 3.297623483e-21
temp = 273

# Lennard-Jones force inputs
r_cut= 7.915 # Angstroms
sigma = 3.166 # Angstroms
epsilon = 0.156 # Kcal/mole
compmethod = 'Naive' # "Naive", "Cellink" or "Cellink_PBC"

# Internal potential parameters
hoh_angle = 103 # degree
oh_len = 1.012  # Angstroms
k_b = 1059.162
tet_eq = 113.24
k_tet = 75.90
# External force parameters
sd_dev = 2
O_charge = -0.834
H_charge = 0.417
epszero = 0.8987
acc_p = 15
k_cut = 2 * acc_p / box_len


# Time Integration parameters (femtoseconds)
timesteps = 1000
timespan= (0, 10)
grid = np.linspace (timespan[0], timespan[1], timesteps)

# Iteration period for saving data into hdf5 files
save_data_itr = 20

'''
visualization parameters :
'''
hdf5_file_name = "data.hdf5"
x_upper = 10
y_upper = 10
z_upper = 10
update_interval_animation = 20
