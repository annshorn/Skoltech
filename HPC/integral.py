from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#https://www.sharcnet.ca/help/images/4/4b/Python_mpi_gis.pdf

def trapezoidal(f, a, b, n, h):
    #h = (b-a)/float(n)
    s = 0.5*(f(a) + f(b))
    for i in range(1,n,1):
        s = s + f(a + i*h)
    return h*s

def func(x):
    return 1/(np.sqrt(1 + x**2))

def Get_data(rank, size, comm):
    a=None
    b=None
    n=None
    if rank == 0:
        a = 5
        b = 7
        n = 10000
        
    a=comm.bcast(a)
    b=comm.bcast(b)
    n=comm.bcast(n)
    return a,b,n

a,b,n = Get_data(rank, size, comm) # process 0 will read data
h = (b-a)/n # h is the same for all processes
local_n = int(n/size) # So is the number of trapezoids

print('local_n:', local_n)


local_a = a + rank*local_n*h
local_b = local_a + local_n*h
integral = trapezoidal(func, local_a, local_b, local_n, h)

# Add up the integrals calculated by each process
total=comm.reduce(integral)

if (rank == 0):
    print("With n=",n,", trapezoids, ")
    print("integral from",a,"to",b,"=",total)

MPI.Finalize