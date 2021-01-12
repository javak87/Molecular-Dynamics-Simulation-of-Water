import numpy as np
import h5py

if __name__=="__main__":
    #Create some Data
    N = 10           #Number of Molecular
    M = 3*N          #Number of Atom .. 1x Molecular = 1x O 2x H
    Timestep = 4     #Integration steps

    #r = np.random.rand(M,3)     #position x,y,z of any Atom
    v = np.random.rand(M,3)     #velocity x,y,z of any Atom
    ptcl = np.arange(1,M+1)     # Partical Number


    #Implementation of Code
    hf = h5py.File('data.h5', 'w')           #Create File to save the data

    for i in range(1,Timestep+1):
        r = np.random.rand(M,3)                              #Create new "positons" for new Timestep
        group1 = hf.create_group('Timestep_{0}'.format(i))   #Create a group for 1x Timestep
        group1.create_dataset('Positions', data = r)
        group1.create_dataset('Velocity', data = v)
        group1.create_dataset('Partical', data = ptcl)
    
    #print(group1.values())
    for i in range(1,Timestep+1):
        dataposition1 = list(group1.get('/Timestep_{0}/Positions'.format(i)))
        print(dataposition1[:])
        print(dataposition1[0])
        print(dataposition1[0][0])


    #dataposition1 = list(group1.get('/Timestep_1/Positions'))
    #print(dataposition1[:])
    #print(dataposition1[0])
    #print(dataposition1[0][0])

    #dataposition4 = list(group1.get('/Timestep_4/Positions'))
    #print(dataposition4[:])
    #print(dataposition4[0])
    #print(dataposition4[0][0])

"""
    with h5py.File('read.hdf5', 'w') as f:
        f.create_dataset('position' , data = r)
        f.create_dataset('velocity' , data = v)
        f.create_dataset('partical' , data = ptcl)

    with h5py.File('read.hdf5', 'r') as f:
        d1 = f['position']
        d2 = f['velocity']

    print(d1)
    print(d2)


    h5_file = h5py.File('level_h5py.h5', mode = 'w')
    group = h5_file.create_group('level')

    my_dtype = numpy.dtype([('Partical N' , 'i'),('Partical' , 'i'), ('rx','d'), ('ry','d'), ('rz','d')])
    dset = group.create_dataset('raw_data', (M,),dtype= my_dtype)

    for i in range(M):
        row = dset[i]
        row ['Partical N'] = 1
        row ['Partical'] = 2
        row ['rx'] = 3.0
        row ['ry'] = 4.0
        row ['rz'] = 5.0
        dset[i] = row
    """