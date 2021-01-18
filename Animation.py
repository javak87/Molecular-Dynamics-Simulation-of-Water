# 16.01.2021
# Author: Florian Anders

import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
import numpy as np
from FileOperation import FileOperation as FileIO


class Animation:

    @staticmethod
    def update_graph(step: int):

        """
        This method updates the animated scatter points

        :param step: The step is the simulated timestep
        :return:
        """

        graph._offsets3d = \
            (
                position_array[step][:][..., 0], position_array[step][:][..., 1], position_array[step][:][..., 2]
            )

        # This is to update the animation

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

    hdf5_file_name = "data.hdf5"


    timesteps = 1000
    time_interval = 5
    molecule_count= int (FileIO.return_molecule_count('data.hdf5'))

    # file creation for saving data
    # NOTE: the created .txt file is for testing purposes only !


    #FileIO.save_data(timesteps, time_interval, molecule_count, "data.hdf5")
    #FileIO.write_hdf5_txt('data.hdf5')

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
