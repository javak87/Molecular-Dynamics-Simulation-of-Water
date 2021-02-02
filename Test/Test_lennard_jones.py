import numpy as np
import numpy.testing as npt
import math
from lattice_config import LatticeConfig
from lennard_jones import LennardJones


def apply_lennard_jones () :
    '''
    this test provides a validation for lennard jones force.
    Newton's third law has been employed to test lennard jones.
    '''
    # create a lattice configuration for atoms positions
    box_len=1000 # Angstroms
    intmolecdist = 250 # Angstroms
    hoh_angle = 103 # degree
    oh_len = 0.97  # Angstroms
    lattice_object = LatticeConfig (intmolecdist, hoh_angle, oh_len, box_len)
    postate = lattice_object()

    sigma = 3.166 # Angstroms
    epsilon = 0.156 # Kcal/mole
    r_cut= 500 # Angstroms

    # Compute Lennard Jones with diffrent methods
    force_object = LennardJones(sigma, epsilon, 'Cellink_PBC', r_cut, box_len)
    lj_force_cell_linked = force_object (postate)
    lj_force_periodic = force_object (postate)
    lj_force_naive = force_object (postate)

    # Test Newton's third law
    npt.assert_almost_equal(np.sum(lj_force_cell_linked, axis=0), [1e-20, 1e-20, 1e-20], decimal=20)
    npt.assert_almost_equal(np.sum(lj_force_periodic, axis=0), [1e-20, 1e-20, 1e-20], decimal=20)
    npt.assert_almost_equal(np.sum(lj_force_naive, axis=0), [1e-20, 1e-20, 1e-20], decimal=20)


if __name__=="__main__":
    apply_lennard_jones ()
