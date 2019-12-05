from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

r = np.linspace(0,4,10000)
r = np.array_split(r, size)[rank]

#print('rank:', rank, 'length of r:', len(r))

main_array = []

if rank != 0:
    X = []
    for r_ in r:
        x = np.random.random()
        for i in range(1000):
            x=(r_*x)*(1-x)
        X.append(x)
    comm.send(X, dest = 0, tag = 11)

elif rank == 0:
    for r_ in r:
        x = np.random.random()
        for i in range(1000):
            x=(r_*x)*(1-x)
        main_array.append(x)

    for source in range(1,size):
        Y = comm.recv(source = source, tag = 11)
        for i in Y:
            #print('i =', i)
            main_array.append(i)

else:
    print("!!!")

