import numpy as np
'''
These are the input for molecular dynamic of water.
'''

'''
Simulation Parameters :
'''
# position Initialization parameters
box_len=15 # Angstroms
intmolecdist = 3 # Angstroms 3 for water

# velocity Initialization parameters
H_mass = 1.00794
O_mass = 16
Kb = 0.001985875
temp = 273

# Lennard-Jones force inputs
r_cut= 1 # Angstroms
sigma = 3.166 # Angstroms
epsilon = 0.156 # Kcal/mole
compmethod = 'Naive' # "Naive", "Cellink" or "Cellink_PBC"

# Internal potential parameters
hoh_angle = 103 # degree
oh_len = 0.97  # Angstroms
k_b=3.5
tet_eq=52
k_tet=1.2

# Time Integration parameters
timesteps = 1000
timespan= (0, 10)
grid = np.linspace (timespan[0], timespan[1], timesteps)

# Iteration period for saving data into hdf5 files
save_data_itr = 5

'''
visualization parameters :
'''
hdf5_file_name = "data.hdf5"
x_upper = 10
y_upper = 10
z_upper = 10
update_interval_animation = 1
