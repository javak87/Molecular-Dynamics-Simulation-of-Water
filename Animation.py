# 16.01.2021
# Author: Florian Anders

import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
import numpy as np
import math
from FileOperation import FileOperation as FileIO


class Animation:

    @staticmethod
    def update_graph(step: int, maxstep: int, graph, array: np.ndarray, t_interval: int, axis, update_interval: int):

        """
        This method updates the animated scatter points

        :param step: The step is the simulated timestep
        :return:
        """

        graph._offsets3d = \
            (
                array[step][:][..., 0], array[step][:][..., 1], array[step][:][..., 2]
            )

        axis.set_title\
            (
                "3D Animation of scatters representing atoms\n\nTimestep: {0} / {1} fs, "
                "Time Interval: {2} fs\n\nUpdate Interval For Animation: {3} ms".format(step * t_interval, maxstep, t_interval, update_interval))

    @staticmethod
    def define_colour_array(molecule_count: int, rgb_value_array_oxygen: np.ndarray, rgb_value_array_hydogen: np.ndarray):

        """
        This method assigns each atom type a specific colour.

        :param molecule_count: number of molecules
        :param rgb_value_array_oxygen: Array containing the desired RGB-values in the simulation for the oxygen atom
        :param rgb_value_array_hydogen: Array containing the desired RGB-values in the simulation for the hydrogen atom
        :return: Numpy-array of atom-specific colours
        """

        colour_array = np.zeros(shape=(molecule_count * 3, 3))

        for i in range(molecule_count * 3):

            if i % 3 == 0 or i == 0:

                colour_array[i, 0] = rgb_value_array_oxygen[0]
                colour_array[i, 1] = rgb_value_array_oxygen[1]
                colour_array[i, 2] = rgb_value_array_oxygen[2]

            else:

                colour_array[i, 0] = rgb_value_array_hydogen[0]
                colour_array[i, 1] = rgb_value_array_hydogen[1]
                colour_array[i, 2] = rgb_value_array_hydogen[2]

        return colour_array

    @staticmethod
    def define_atom_size_array(molecule_count: int, size_of_hydrogen: float, size_ratio_oxygen_hydrogen: float):

        """
        This method assigns each atom type a specific radius shown in the animation.

        :param molecule_count: Number of molecules
        :param size_of_hydrogen: This is an arbitrary radius for the animated sphere representing hydrogen atoms
        :param size_ratio_oxygen_hydrogen: This ratio determines, how much larger an oxygen atom is compared to hydrogen
        :return: Numpy-array of atom-specific radius-values
        """

        size_array = np.zeros(shape=(molecule_count * 3))

        for i in range(molecule_count * 3):

            if i % 3 == 0 or i == 0:

                size_array[i] = size_of_hydrogen * size_ratio_oxygen_hydrogen

            else:

                size_array[i] = size_of_hydrogen

        return size_array


if __name__ == "__main__":

    # defining initial values

    hdf5_file_name_one = "data_3000.hdf5"
    hdf5_file_name_two = "data_1000.hdf5"
    hdf_file_t_fine = 'data.hdf5'

    timesteps_one = 3000
    timesteps_two = 1000
    timesteps_fine = 10000
    time_interval = 10
    molecule_count_one = int(FileIO.return_molecule_count('data_3000.hdf5'))
    molecule_count_two = int(FileIO.return_molecule_count('data_1000.hdf5'))
    molecule_count_fine = int(FileIO.return_molecule_count('data.hdf5'))
    # file creation for saving data
    # NOTE: the created .txt file is for testing purposes only !


    #FileIO.save_data(timesteps, time_interval, molecule_count, "data.hdf5")
    #FileIO.write_hdf5_txt('data.hdf5')

    # accessing the saved data and assigning the positional values to a numpy array

    position_array_one = np.array(FileIO.extract_data_to_np_array(timesteps_one, time_interval, molecule_count_one, hdf5_file_name_one)[0])
    velocity_array_one = np.array(FileIO.extract_data_to_np_array(timesteps_one, time_interval, molecule_count_one, hdf5_file_name_one)[1])

    position_array_two = np.array(FileIO.extract_data_to_np_array(timesteps_two, time_interval, molecule_count_two, hdf5_file_name_two)[0])
    velocity_array_two = np.array(FileIO.extract_data_to_np_array(timesteps_two, time_interval, molecule_count_two, hdf5_file_name_two)[1])

    position_array_fine = np.array(FileIO.extract_data_to_np_array(timesteps_fine, time_interval, molecule_count_fine, hdf_file_t_fine)[0])
    velocity_array_fine = np.array(FileIO.extract_data_to_np_array(timesteps_fine, time_interval, molecule_count_fine, hdf_file_t_fine)[1])

    # for animation, the atom-specific colours are now defined

    rgb_array_oxygen = np.array([255, 88, 32])
    rgb_array_hydrogen = np.array([0, 96, 255])
    colours = Animation.define_colour_array(molecule_count_fine, rgb_array_oxygen, rgb_array_hydrogen)

    # for animation, the atom-specific sizes (based roughly on VDW-radii) are now defined

    sizes_of_atoms = Animation.define_atom_size_array(molecule_count_fine, 49, 2)

    # the plot and figure are created, using the data from the hdf5 file

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    title = ax.set_title("3D Animation of scatters representing atoms; \n\nTimestep: {0}, Time Interval: {1}".format(0, time_interval))
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

    x_position_range = (np.amin(position_array_fine[:][:][..., 0]), np.amax(position_array_fine[:][:][..., 0]))
    y_position_range = (np.amin(position_array_fine[:][:][..., 1]), np.amax(position_array_fine[:][:][..., 1]))
    z_position_range = (np.amin(position_array_fine[:][:][..., 2]), np.amax(position_array_fine[:][:][..., 2]))
    ax.set_xlim(x_position_range)
    ax.set_ylim(y_position_range)
    ax.set_zlim(0, 5)

    # create the graph with scatter points and animate the simulation using a repeated call to the update_graph method

    graph = ax.scatter(position_array_fine[0][:][..., 0], position_array_fine[0][:][..., 1], position_array_fine[0][:][..., 2],
                       c=colours/255, s=sizes_of_atoms)

    update_interval_animation = 10 # in milliseconds

    anim_func = animation.FuncAnimation\
        (
            fig,
            Animation.update_graph,
            timesteps_fine // time_interval,
            fargs=
            (
                timesteps_fine,
                position_array_fine,
                time_interval,
                ax,
                update_interval_animation
            ),
            interval=update_interval_animation,
            blit=False
        )

    # finally show the animation

    plt.show()

    # show the trajectory of the first hydrogen atom in x - direction

    fig = plt.figure()
    fig.add_subplot()

    plt.plot(position_array_fine[:][:, 1][..., 0], velocity_array_fine[:][:, 1][..., 0])
    plt.xlabel('X - coordinate')
    plt.ylabel('Velocity in X - direction')

    plt.show()

    # show the trajectory of the first hydrogen atom in y - direction

    plt.plot(position_array_fine[:][:, 1][..., 1], velocity_array_fine[:][:, 1][..., 1])
    plt.xlabel('Y - coordinate')
    plt.ylabel('Velocity in Y - direction')

    plt.show()

    # show the trajectory of the first hydrogen atom in z - direction

    plt.plot(position_array_fine[:][:, 1][..., 2], velocity_array_fine[:][:, 1][..., 2])
    plt.xlabel('Z - coordinate')
    plt.ylabel('Velocity in Z - direction')

    plt.show()
