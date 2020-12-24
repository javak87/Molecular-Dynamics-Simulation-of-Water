import numpy as np
import math

class CellLinked ():

    def __init__ (self, box_len: float, r_cut: float) :
        """
        This class initialize with cubic simulation box (L*L*L).
        Here, L=box_len and r_cut is a cut-off radius.
        no_cells is number of cell per axis
        """
        self.box_len = box_len
        self.r_cut = r_cut
        self.no_cells = math.floor (self.box_len/self.r_cut)

    def __call__ (self, postate: np.ndarray) :

        """ 
            Compute the head and list array using simulation box length, cut-off radius, and the position state.

            Parameters:
                postate (np.ndarray):       position of particles

            Returns
                head_arr (list):   head array in the linked list (head_arr.shape==number of cells)
                list_arr (list):   list array in the linked list (list_arr.shape==number of particles)
        """

        # Initialize a Head array with -1, size = the total number of cells
        head_arr = [-1] * self.no_cells*self.no_cells*self.no_cells

        # Initialize a list array with -1, size = the total number of particles
        list_arr = [-1] * postate.shape[0]

        for i in range (0, postate.shape[0]) :

            # calculate cell index of each particles based on the particle position
            index = self._cellindex (postate[i,:])

            # calculate list array
            list_arr [i] = head_arr [index]

            # calculate head array         
            head_arr [index] = i

        return head_arr, list_arr


            


    def _cellindex (self, postate) :

        """ Calculate the cell index based on the i, j, and k index
        """
        i = math.floor (postate[0]/self.r_cut)
        j = math.floor (postate[1]/self.r_cut)
        k = math.floor (postate[2]/self.r_cut)
        return int (k+j*self.no_cells+i*self.no_cells*self.no_cells)

    def _find_neighbor_index (self, cellind) :

        """
        calculate neighbor cell index based on the cell index
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

    def _find_neighbor_cells (self, head_arr) :

        """
        calculate neighbor cells based on head array
        """

        cells = []
        for j in range (0, len (head_arr)) :           
            neighbor_cell_inx = self._find_neighbor_index (j)           
            cells.append(neighbor_cell_inx)

        return cells

if __name__=="__main__":

    box_len=3
    r_cut= 1
    postate= np.random.randint(box_len, size=(20, 3))
    print ( postate)
    model = CellLinked (box_len,  r_cut)
    head_arr, list_arr = model (postate)
    cells = model._find_neighbor_cells(head_arr)
    print (cells)
    print ("the head array is : \n ", head_arr)
    print ("list array is : \n ", len (list_arr))

    
