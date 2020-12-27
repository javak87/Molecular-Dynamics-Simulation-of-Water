import numpy as np
import math

class CellLinked ():

    def __init__ (self, box_len: float, r_cut: float, postate : np.ndarray) :
        """
        Initialize parameters for identified neighbor cells and particles inside these cells

            Parameters:
                box_len (float) : cube simulation box dimension (box_len*box_len*box_len)
                r_cut (float) : cut-off radius
                no_cells (float) : number of cell per axis
                postate (np.ndarray): position of particles in 3D
        """
        self.box_len = box_len
        self.r_cut = r_cut
        self.no_cells = math.floor (self.box_len/self.r_cut)
        self.postate = postate

    def __call__ (self) -> tuple:

        """ 
            Compute the head and list array using simulation box length, cut-off radius, and the position state.

            Returns
                head_arr (list):   head array in the linked list (head_arr.shape==number of cells)
                list_arr (list):   list array in the linked list (list_arr.shape==number of particles)
        """

        # Initialize a Head array with -1, size = the total number of cells
        head_arr = [-1] * self.no_cells*self.no_cells*self.no_cells

        # Initialize a list array with -1, size = the total number of particles
        list_arr = [-1] * self.postate.shape[0]

        print (list_arr)

        for i in range (0, self.postate.shape[0]) :
            print (i)

            # calculate cell index of each particles based on the particle position
            index = self._cellindex (self.postate[i,:])

            # calculate list array
            list_arr [i] = head_arr [index]

            # calculate head array         
            head_arr [index] = i

        return head_arr, list_arr


            


    def _cellindex (self, one_particle_pos : np.ndarray) -> int :

        """ Calculate the cell index based on the i, j, and k index

                Parameter :
                    one_particle_pos (np.ndarray) : position of 1 particle (one_particle_pos.shape = (1,3))
                
                Return :
                    cell's index (int) : cell index for each particles 

        """
        i = math.floor (one_particle_pos[0]/self.r_cut)
        j = math.floor (one_particle_pos[1]/self.r_cut)
        k = math.floor (one_particle_pos[2]/self.r_cut)
        return int (k+j*self.no_cells+i*self.no_cells*self.no_cells)

    def _find_neighbor_index (self, cellind: int) -> list:

        """
        calculate neighbor cell index based on the cell index
            Parameter :
                cellind (int) : cell index which is camputed by _cellindex method

            Return :
                neighbor_cell_inx (list) : neighbor cell indexes by the given cell index

        """
        #  calculate i, j, and k index based on the cell index

        i_counter = math.floor (cellind/self.no_cells**2)
        j_counter = math.floor ((cellind - i_counter*self.no_cells**2)/self.no_cells)
        k_counter = math.floor (cellind - j_counter*self.no_cells - i_counter*self.no_cells**2)
      

        # calculate neighbor cell index for each cell index
        index = [-1,0,1]
        neighbor_cell_inx = []
        for ii in index :
            for jj in index :
                for kk in index :
                    if 0 <= ii + i_counter < self.no_cells and 0 <= jj + j_counter < self.no_cells and 0 <= kk + k_counter < self.no_cells :
                        new_index = (kk + k_counter) + (jj + j_counter) *self.no_cells + (ii + i_counter) *self.no_cells*self.no_cells
                        neighbor_cell_inx.append(new_index)
        return neighbor_cell_inx

    def _find_neighbor_cells (self, head_arr: list) -> list:

        """
        calculate neighbor cells based on head array

            Parameter :
                head_arr (list) : a list of head array

            Return :
                neighbor_cells (list) : neighbor cell indexes
        """

        neighbor_cell_indexes = []
        for j in range (0, len (head_arr)) :           
            neighbor_cell_inx = self._find_neighbor_index (j)           
            neighbor_cell_indexes.append(neighbor_cell_inx)

        return neighbor_cell_indexes
    
    def _find_particles_inside_neighbor_cells (self, list_arr : list, head_arr : list, neighbor_cell_indexes : list) -> np.ndarray:
    
        """
        find particles in each cell and neighbor cells are identified
            Parameters :
                list_arr (list) : list array
                head_arr (list) : head array
                neighbor_cell_indexes (list) : neighbor cells indexes

            Return :
                temporary_state (np.ndarray) : state of all particles positioned on the given neighbor_cell_indexes
        """
        new_head_arr =[]
        for i in neighbor_cell_indexes :
            new_head_arr.append (head_arr[i])
        temporary_state = np.zeros((1,3))

        for ii in new_head_arr:
            if ii > -1 :
                cell_state = np.zeros((1,3))
                cell_state = np.concatenate((cell_state, np.array ([self.postate[ii]])))
                while  list_arr[ii] != -1 :
                    cell_state = np.concatenate((cell_state, np.array ([self.postate[list_arr[ii]]])))
                    ii = list_arr[ii]

                cell_state = np.flip(cell_state[1:, :], 0)
                temporary_state = np.concatenate((temporary_state, cell_state))
        
        
        return temporary_state[1:, :]




if __name__=="__main__":

    box_len=5
    r_cut= 1
    postate = box_len * np.random.random_sample((10, 3))
    print (postate)
    model = CellLinked (box_len,  r_cut, postate)
    head_arr, list_arr = model ()
    print (len(list_arr))
    print (len(head_arr))
    neighbor_cells = model._find_neighbor_cells(head_arr)
    neighbor_cells = neighbor_cells[13]
    temporary_state = model._find_particles_inside_neighbor_cells (list_arr, head_arr, neighbor_cells)

    print (temporary_state.shape)
    print (temporary_state)
    #mask = np.isin(postate, temporary_state)
    #assert np.all(mask == True) == True, 'particles in neighbor cells has been indetified wrongly'

    print ("the head array is : \n ", head_arr)
    print ("list array is : \n ", list_arr)

    
