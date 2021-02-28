# 12.01.2021
# Author: Florian Anders

import numpy as np
import h5py
import sys
import math


class FileOperation:

    def __init__(self):

        pass

    @staticmethod
    def save_data(timesteps: int, time_interval: int, molecule_count: int, file_name: str):

        """
        This method saves the data coming from the integrator in a .hdf5 file with groups = timesteps and
        datasets = Position and/or Velocity

        :param timesteps:
        :param time_interval:
        :param molecule_count:
        :param file_name:
        :return:
        """

        atom_count = molecule_count * 3

        hdf5_file = h5py.File(file_name, 'w')

        r = np.random.rand(atom_count, 3)

        # calculated positions from the integrator; right now just random data

        v = np.random.rand(atom_count, 3)

        # calculated velocities from the integrator; right now just random data

        for i in range(0, timesteps + 1):

            # the range was chosen so that we have timestep 0 (initial data) to
            # timestep + 1 (data of the last timestep since in range(start, stop) stop is excluded)

            if i == 0:

                # save the initial positions and velocities, before even calling Integrator methods

                new_group = hdf5_file.create_group('Timestep Identifier Number = 0')
                new_group.create_dataset('Positions', data=r)
                new_group.create_dataset('Velocities', data=v)

            elif i % time_interval == 0:

                # save only the data for every x-th timestep (determined by time_interval)

                new_group = hdf5_file.create_group('Timestep Identifier Number = {0}'.format(i / timesteps))
                new_group.create_dataset('Positions', data=r_new)
                new_group.create_dataset('Velocities', data=v_new)

            r_new = r + i * 0.005 * (-1)**i
            v_new = v + i * 0.001

            # ONLY used for generating retraceable data for visualisation;
            # DO NOT USE with actual, calculated data !

        hdf5_file.close()

    @staticmethod
    def write_hdf5_txt(file_name: str):

        """
        This is just a test function to see how the data is being put out in a text file.
        Should not be used, except for testing purposes
        :param file_name:
        :return: null
        """

        with h5py.File(file_name, 'r') as file:

            np.set_printoptions(threshold=sys.maxsize)

            # necessary to print all entries of a numpy array / everything that uses numpy - style arrays

            file_text = open("{0}.txt".format(file_name), "w")

            for group in file.keys():

                for dataset in file[group]:

                    if dataset == "Positions":

                        data = file["/" + group + "/" + dataset][:]

                        content_string = "Group:\n{0}\nDataset:\n{1}\nData:\n{2}\n\n".format(group, dataset, data)

                        file_text.write(content_string)

                    elif dataset == "Velocities":

                        data = file["/" + group + "/" + dataset][:]

                        content_string = "Group:\n{0}\nDataset:\n{1}\nData:\n{2}\n\n".format(group, dataset, data)

                        file_text.write(content_string)

            file_text.close()
            file.close()

    @staticmethod
    def return_molecule_count(file_name: str):

        with h5py.File(file_name, 'r') as file:

            np.set_printoptions(threshold=sys.maxsize)

            for group in file:

                for dataset in file[group]:

                    return len(file["/" + group + "/" + dataset][:]) / 3

    @staticmethod
    def extract_data_to_np_array(time_steps: int, time_interval: int, molecule_count: int, file_name: str):

        """
        This method extracts data from .hdf5 and stores the data by group and dataset in a numpy-ndarray

        :param file_name: Name of the file to extract data from
        :param molecule_count: Number of simulated molecules (determines the shape of the resulting array)
        :param time_steps: Number of time steps to be simulated
        :param time_interval: Only every x-th data-point should be saved; this is ensured by this paramter
        :return: numpy-ndarrays containing the positional and velocital information
        """
        atom_count = molecule_count*3

        with h5py.File(file_name, 'r') as file:

            np.set_printoptions(threshold=sys.maxsize)

            # necessary to print all entries of a numpy array / everything that uses numpy - style arrays

            array_of_positions = np.zeros(shape=((time_steps // time_interval) + 1, atom_count, 3))
            array_of_velocities = np.zeros(shape=((time_steps // time_interval) + 1, atom_count, 3))

            # time_interval + 1 because of the inclusion of the initial time step data

            i = -1

            for group in file.keys():

                i += 1

                for dataset in file[group]:

                    if dataset == "Positions":

                        position_data = file["/" + group + "/" + dataset][:]

                        array_of_positions[i] = position_data

                    elif dataset == "Velocities":

                        velocity_data = file["/" + group + "/" + dataset][:]

                        array_of_velocities[i] = velocity_data

            file.close()

            return array_of_positions, array_of_velocities


if __name__ == "__main__":

    t_steps = 10
    t_interval = 2

    # t_steps should be a multiple of t_interval, to include the last calculated data;
    # otherwise there will be unnecessary calculations, where the results won't be saved

    mol_count = 15

    FileOperation.save_data(t_steps, t_interval, mol_count, "data.hdf5")
    FileOperation.write_hdf5_txt('data.hdf5')
    print(FileOperation.return_molecule_count('data.hdf5'))

    #pos_array = np.array(FileOperation.extract_data_to_np_array(t_steps, t_interval, "data.hdf5")[0])
    #vel_array = np.array(FileOperation.extract_data_to_np_array(t_steps, t_interval, "data.hdf5")[1])

    # pos_array[0] gives all atom - positions for the first time step

    # pos_array[0][1] gives the position of the second atom ([1]) for the first time step ([0])

    # pos_array[0][1][2] gives the z-coordinate ([2]) of the second atom ([1]) of the first time step ([0])

    # more examples with slicing:

    # pos_array[0][:][..., 0] will give all x-coordinates from all atom positions of the first time step
    # pos_array[0][:][..., 1] will give all y-coordinates from all atom positions of the first time step
    # pos_array[0][:][..., 2] will give all z-coordinates from all atom positions of the first time step

    # the same holds true for the vel_array
