import numpy as np
from scipy.special import erfc
from generate_images import GenerateImages
#from numpy import inf

O_charge = -2
H1_charge = 1.5
H2_charge = 0.5
epszero = 0.8987
box_len = 10
sd_dev = 2
postate = np.random.randint(low=1, high=10, size = (60,3))
bounds = np.array ([10, 10, 10]) 

images_object = GenerateImages (postate, bounds, 10)
periodic_images = images_object.expand_images()


#print (postate)
#print(periodic_images)

# assign oxygen and Hydrogen charge to all atoms
point_charge = np.zeros((postate.shape[0],1))
point_charge[0::3] = O_charge
point_charge[1::3] = H1_charge
point_charge[2::3] = H2_charge

print (point_charge)
#charge_array = np.zeros((periodic_images.shape[0],1), dtype=float)

#charge_array[0:int (periodic_images.shape[0]/3)] = O_charge
#charge_array[int (periodic_images.shape[0]/3):int (2*periodic_images.shape[0]/3)] = H1_charge
#charge_array[int (2*periodic_images.shape[0]/3):int (periodic_images.shape[0])] = H2_charge
#charge_array[0:27:int (periodic_images.shape[0])] = O_charge
charge_array = np.repeat(point_charge, 27).reshape (periodic_images.shape[0],1)
print (charge_array)

#print (charge_array)
#char_O = charge_array*O_charge
#char_H1 = charge_array*H_charge
#char_H2 = charge_array*H_charge
#d = np.vstack ((char_O, char_H1))
#h = np.vstack ((d,char_H2))
#print (h.shape)
#charges = h.reshape(6, 81, 1)
#print (charges)
#firstc = np.array ([char_O], [char_H1])
#second = np.array ([firstc], [char_H2])
##print (first)
charges = point_charge[:, np.newaxis]*charge_array[np.newaxis, :]

#print (point_charge[:, np.newaxis])
#print (charges)


# compute ri-rj-nL vector
periodic_vector =  postate[:, np.newaxis] - periodic_images[np.newaxis, :]
#print (periodic_vector)

norm_periodic_vector = np.linalg.norm (periodic_vector, axis=2).reshape(periodic_vector.shape[0], periodic_vector.shape[1], 1)

#print (norm_periodic_vector.shape)


#xx = charges.reshape(charges.shape[0], charges.shape[1])
#print (xx.shape)
#yy = norm_periodic_vector.reshape(charges.shape[0], charges.shape[1])
#print (yy.shape)

#print (periodic_vector[:, : ,1])

A= periodic_vector/norm_periodic_vector
#print(A)
A[np.isnan(A)]=0
A = A*charges
X= A.sum(axis=1)
#print(X)
print(X.sum(axis=0))
#print (norm_periodic_vector.shape)
#print (A[0][:,0]+A[1][:,0]+A[2][:,0])
#print (A[1][:,0])
#print (A[0][:,1]+A[1][:,1]+A[2][:,1])
#print (A[0][:,2]+A[1][:,2]+A[2][:,2])

#print (sum(A[0][:,0]+A[1][:,0]+A[2][:,0]))
#print (sum(A[0][:,1]+A[1][:,1]+A[2][:,1]))
#print (sum(A[0][:,2]+A[1][:,2]+A[2][:,2]))
#print ('sssss \n', np.sum(np.sum (periodic_vector*charges, axis =1), axis=0))