import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
import numpy as np
from FileOperation import FileOperation as FileIO
from Animation import Animation



hdf5_file_name = "data.hdf5"


timesteps = 3000
time_interval = 5
molecule_count= int (FileIO.return_molecule_count('data.hdf5'))


# accessing the saved data and assigning the positional values to a numpy array

position_array = np.array(FileIO.extract_data_to_np_array(timesteps, time_interval, molecule_count, hdf5_file_name)[0])

# for animation, the atom-specific colours are now defined

rgb_array_oxygen = np.array([180, 225, 0])
rgb_array_hydrogen = np.array([40, 40, 255])
colours = Animation.define_colour_array(molecule_count, rgb_array_oxygen, rgb_array_hydrogen)

# for animation, the atom-specific sizes (based roughly on VDW-radii) are now defined

sizes_of_atoms = Animation.define_atom_size_array(molecule_count, 62.7, 2)

# the plot and figure are created, using the data from the hdf5 file

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title("3D Test")
ax.set_xlabel("X - position")
ax.set_ylabel("Y - position")
ax.set_zlabel("Z - position")

# now the nice slicing of the positional array comes into play;
# the axes of the 3D-plot are scaled based on the maximum and minimum x, y and z - coordinates of all timesteps,
# meaning that the -simulated molecules will always stay in the animation box
# (not to be confused with the simulation box !)

# as an example:
# x_position_range = (np.amax(position_array[:][:][..., 0]), np.amin(position_array[:][:][..., 0]))
# this gives a tuple containing the maximum (np.amax()) and minimum (np.amin()) values
# of all available x-coordinates from all timesteps

x_position_range = (np.amin(position_array[:][:][..., 0]), np.amax(position_array[:][:][..., 0]))
y_position_range = (np.amin(position_array[:][:][..., 1]), np.amax(position_array[:][:][..., 1]))
z_position_range = (np.amin(position_array[:][:][..., 2]), np.amax(position_array[:][:][..., 2]))
ax.set_xlim(0, 1000)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)

# create the graph with scatter points and animate the simulation using a repeated call to the update_graph method

graph = ax.scatter(position_array[0][:][..., 0], position_array[0][:][..., 1], position_array[0][:][..., 2],
                    c=colours/255, s=sizes_of_atoms)

anim_func = animation.FuncAnimation(fig, Animation.update_graph, timesteps // time_interval, interval=1, blit=False)

# finally show the animation

plt.show()