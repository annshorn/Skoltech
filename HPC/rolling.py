from mpi4py import MPI
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import memory_profiler as mem_profile
%load_ext memory_profiler
%memit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

img = Image.open('dog.jpeg')
img_array = np.asarray(img)[:,:,0]

print('the real shape of image:',img_array.shape)
image = np.array_split(img_array,size,axis=1)[rank]
print('rank:', rank, 'the shape for rank:', image.shape)

step = 50

if rank == size-1:
    x = image
    for i in range(step):
        x = np.roll(x, -1, axis=1)
        xrec = comm.recv(source = 0)
        #print('shape of the columns:', xrec.reshape(-1,1).shape)
        comm.send(x[:,-1], dest = 0)
        x = np.delete(x,-1,axis=1)
        x = np.hstack((x,xrec.reshape(-1,1)))
            
if rank == 0:
    x = image
    for i in range(step):
        x = np.roll(x, -1, axis=1)
        comm.send(x[:,-1], dest = size-1)
        x = np.delete(x,-1,axis=1)
        xrec = comm.recv(source = size-1)
        x = np.hstack((x,xrec.reshape(-1,1)))
        
gathedata =  comm.gather(x,root=0)
if not rank:
    new_im = np.hstack(gathedata)
    plt.imsave('jifs/colmwise.png',new_im)



# if rank == size-1:
#     x = image
#     for i in range(200):
#         x = np.roll(x, -1, axis=1)
#         #stacked = x
#         stacked = np.concatenate([x,img_array[:,:i]], -1)
#         comm.send(stacked, dest = 0)
#         x = np.delete(x,-1,axis=1)
 
        
# elif rank == 0:
#     x = image
#     #print(x.shape)
#     for i in range(200):
#         X = []
#         x = np.roll(x, -1, axis=1)
#         X.append(x)
#         #x = np.delete(x,-1,axis=1)
#         for source in range(1,size):
#             C = comm.recv(source = source)
#             X.append(C)
#             final_image = np.concatenate(X, -1)
#             img = Image.fromarray(final_image)
#             img.save('jifs/roll' + str(i) + '.png')
        

# if rank != 0 and rank != size-1:
#     #x = image
#     start = int(img_array.shape[1]*rank/size)
#     finish = int(img_array.shape[1]*(rank+1)/size)
#     img_array[:,start:finish]
#     for i in range(200):
#         x = img_array[:,start+i:finish+i]#np.roll(x, -1, axis=1)
#         #img = Image.fromarray(x)
#         comm.send(x, dest = 0, tag = 11)
#         #x = np.delete(x,-1,axis=1)
#         #img.save('doggy_proverka1' + str(i) + '.png')

# elif rank == size-1:
#     #x = image
#     start = int(img_array.shape[1]*rank/size)
#     finish = int(img_array.shape[1]*(rank+1)/size)
#     img_array[:,start:finish]
#     for i in range(200):
#         x = img_array[:,start+i:finish-i]#np.roll(x, -1, axis=1)
#         stacked = x
#         #stacked = np.concatenate([x,img_array[:,:i]], -1)
#         comm.send(stacked, dest = 0, tag = 11)
#         #x = np.delete(x,-1,axis=1)
        
# elif rank == 0:
# #     x = image
#     start = int(img_array.shape[1]*rank/size)
#     finish = int(img_array.shape[1]*(rank+1)/size)
#     for i in range(200):
#         X = []
#         x = img_array[:,start+i:finish+i]
#         X.append(x)
#         #x = np.delete(x,-1,axis=1)
#         for source in range(1,size):
#             C = comm.recv(source = source, tag = 11)
#             X.append(C)
#             final_image = np.concatenate(X, -1)
#             img = Image.fromarray(final_image)
#             img.save('jifs/roll' + str(i) + '.png')
            
    

