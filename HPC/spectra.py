from mpi4py import MPI
import numpy as np

from numpy import exp as exp
from numpy import fft 
from numpy import pi as pi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def my_imshow(x, y, z, 
              title, 
              xlabel, 
              ylabel,
              name = '1',
              showplt = True,
              grid_active = False, fig_x_size = 15, fig_y_size = 10, font_param = 20):
    
    plt.figure(figsize=(fig_x_size, fig_y_size))
    plt.imshow(z, aspect='auto', 
               origin='lower', 
               extent=[min(x)/2/pi, max(x)/2/pi, y[0], 2 * w[int(len(x)/2)-1]])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=font_param)
    plt.title(title, fontsize = font_param * 1.3)
    plt.xlabel(xlabel, fontsize = font_param)
    plt.ylabel(ylabel, fontsize = font_param)
    plt.xticks(fontsize = font_param)
    plt.yticks(fontsize = font_param)
    plt.grid(grid_active)
    plt.ylim(0, 10)
    plt.savefig('Spectogram'+name+'.png')
    if showplt:
        plt.show()
    

# signal consits of wave packets of three harmonic signals
def form_signal(n_timestamps = 4096):
    t=np.linspace(-20*2*np.pi, 20*2*np.pi, n_timestamps)
    y=np.sin(t)*exp(-t**2/2/20**2) #generate first  wave packets of harmonic signal
    y=y+np.sin(3*t)*exp(-(t-5*2*np.pi)**2/2/20**2) #add      second wave packets of harmonic signal
    y=y+np.sin(5*t)*exp(-(t-10*2*np.pi)**2/2/10**2) #add      third  wave packets of harmonic signal
    #frequency=4 
    #time_shift=7 
    y=y+np.sin(7*t)*exp(-(t-4*2*np.pi)**2/2/10**2)  #add      fouth  wave packets of harmonic signal
    return t, y

def window_function(t, window_position, window_width):
    return exp(- (t - window_position) ** 2 / 2 / window_width ** 2)


t,y = form_signal()
sp=fft.fft(y)
w=fft.fftfreq(len(y), d=(t[1]-t[0])/2/pi)

kappa = 1
window_width_given = kappa * 2 * np.pi
nwindowsteps_given = 1000
window_width = window_width_given
nwindowsteps = nwindowsteps_given

x=np.linspace(-20 * 2 * pi, 20 * 2 * np.pi, nwindowsteps)
t_window_positions = np.array_split(x,size)[rank]
specgram = np.empty([len(t), len(t_window_positions)])

for i,t_window_position in enumerate(t_window_positions):
    y_window=y * window_function(t, t_window_position, window_width)
    #plot(y_window)
    specgram[:,i]=abs(fft.fft(y_window))
    
gatherdata = comm.gather(specgram,root=0)

if not rank:
    spect = np.hstack(gatherdata)
#     my_imshow(t, w, spect, title = "Specgram", 
#           xlabel = "t, cycles", 
#           ylabel = "Frequency, arb. units")
#     print('rank',rank,)
#     print('recieved buffer for \t {}  :'.format(spect.shape))
