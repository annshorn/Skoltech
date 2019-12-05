from mpi4py import MPI
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import sys
from scipy.signal import correlate2d as conv2d

comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank()

def numpy_update(alive_map):
    # Count neighbours with convolution
    conv_kernel = np.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]])
    
    num_alive_neighbors = conv2d(alive_map, conv_kernel, mode='same')
    # Apply game rules
    born = np.logical_and(num_alive_neighbors == 3, alive_map == 0)
    survived = np.logical_and(np.isin(num_alive_neighbors, [2,3]), alive_map == 1)

    np.copyto(alive_map, np.logical_or(born, survived))

# initialize game field
alive_map = np.random.choice([0, 1], size=(100, 100))

x = np.array_split(alive_map,size,axis=1)[rank]
step = 100

for i in range(step):
    if rank == 0:
        #for i in range(step):
            numpy_update(x)
            comm.send(x[:,-1], dest = size-1)
            x = np.delete(x,-1,axis=1)
            xrec = comm.recv(source = size-1)
            x = np.hstack((x,xrec.reshape(-1,1)))

    if rank == size - 1:
        #for i in range(step):
            numpy_update(x)
            comm.send(x[:,-1], dest = 0)
            x = np.delete(x,-1,axis=1)
            xrec = comm.recv(source = 0)
            #print('ddd =', xrec.reshape(-1).shape)
            x = np.hstack((x,xrec.reshape(-1,1)))
            #print(x.shape)

    gathedata =  comm.gather(x,root=0)
    if not rank:
        new_im = np.hstack(gathedata)
        plt.imsave('game/game' + str(i) + '.png',new_im)
    
#     for _ in range(10):
#         numpy_update(new_im)
    #plt.imsave('game/game' + '.png',new_im)
  

