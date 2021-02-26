import numpy as np

class PeriodicBoundary :

    def __init__(self, box_len: float, postate: np.ndarray) :
        '''
        initialize simulation box length and position state of all atoms
        Parameters:
                   box_len (float): simulation box length (Angstrom)
                   postate (np.ndarray): position of all atoms
        '''

        self.box_len = box_len
        self.postate = postate

    def __call__ (self) :
        '''
        This class applied periodic boundary conditions to a simulation box.
        To make sure all atoms remain in the simulation box, using periodic boundary conditions is necessary.
        Each water molecule includes 1 oxygen atom and two hydrogen atoms.
        In this class, when all atoms of one water molecule cross the border, the periodic boundary is applied.
        '''

        return self.periodic_boundary()


    def periodic_boundary (self) :

        # recoganize the elements that cross the boundary
        out_array = self.postate - np.where (np.logical_or(self.postate > self.box_len, self.postate < 0), self.postate , 0)

        # summation of out_array in x, y, and z coordinates each molecule to make sure all atoms of one molecule
        # cross the boundary
        summ_coordin = out_array.reshape(-1,self.postate.shape[1],out_array.shape[-1]).sum(1)

        # create a boolean np.array to show which molecule cross the boundary (all atoms of one molecule should cross the border)
        condition = summ_coordin == 0
        booli = np.repeat(condition, repeats = int(self.postate.shape[0]/3)*[3], axis=0)
         

        # extract elements that cross bondary
        out_bond_array = self.postate[booli]

        # Apply periodic boundary
        replace_elem = np.abs(self.box_len - np.abs(out_bond_array))

        self.postate[booli] = replace_elem

        return self.postate

if __name__=="__main__":


    box_len = 3
    postate = np.random.randint(low=-4, high=4, size = (15,3))
    print ('postate \n', postate)
    periodic_object = PeriodicBoundary (box_len, postate)
    periodic = periodic_object ()

    #print ('postate \n', postate)
    print ('periodic \n', periodic)


        


        
