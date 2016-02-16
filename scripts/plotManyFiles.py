#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
import matplotlib.animation as animation

def particleplotting(filenames,tracerfile, mode):
    """Quick and simple plotting of PARCELS trajectories"""

    filename = filenames[0]
    pfile=Dataset(filename,'r')
    lon = np.empty((np.size(pfile.variables['lon'],0),np.size(pfile.variables['lon'],1),len(filenames)))
    lat = np.empty((np.size(pfile.variables['lat'],0),np.size(pfile.variables['lat'],1),len(filenames)))
    z = np.empty((np.size(pfile.variables['z'],0),np.size(pfile.variables['z'],1),len(filenames)))
    fig, ax = plt.subplots()
    labels = ('Analytical', 'RK4', 'EE')    #If you call them in this order...
    for i in range(len(filenames)):
        filename = filenames[i]
        pfile=Dataset(filename,'r')
        lon[:,:,i] = pfile.variables['lon'][:,:]
        lat[:,:,i] = pfile.variables['lat'][:,:]
        z[:,:,i] = pfile.variables['z'][:,:]
#        lon = pfile.variables['lon']
#        lat = pfile.variables['lat']
#        z = pfile.variables['z']

#        if tracerfile != 'none':
#            tfile = Dataset(tracerfile,'r')
#            X = tfile.variables['x']
#            Y = tfile.variables['y']
#            P = tfile.variables['P']
#            plt.contourf(np.squeeze(X),np.squeeze(Y),np.squeeze(P))
#
#        if mode == '3d':
#            ax = fig.gca(projection='3d')
#            for p in range(len(lon)):
#                ax.plot(lon[p,:],lat[p,:],z[p,:],'.-')
#            ax.set_xlabel('Longitude')
#            ax.set_ylabel('Latitude')
#            ax.set_zlabel('Depth')
        if mode == '2d':
            plt.plot(np.transpose(lon[:,:,i]),np.transpose(lat[:,:,i]),'.-')
#            plt.plot(np.transpose(lon),np.transpose(lat),'.-')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
#        elif mode =='movie2d':
#            line, = ax.plot(lon[:,0], lat[:,0],'ow')
#            if tracerfile == 'none': # need to set ax limits
#                plt.axis((np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)))
#
#            def animate(i):
#                line.set_xdata(lon[:,i])
#                line.set_ydata(lat[:,i])
#                return line,
#
#            ani = animation.FuncAnimation(fig, animate, np.arange(1, lon.shape[1]),
#                                          interval=100, blit=False)
#            plt.show()
    if len(filenames) > 1:
        fig2, ax2 = plt.subplots()
        for i in range(1,len(filenames)):
            plt.plot(np.sqrt((np.transpose(lon[:,:,i]) - np.transpose(lon[:,:,0]))\
                    ** 2 + (np.transpose(lat[:,:,i]) - np.transpose(lat[:,:,0])) ** 2),\
                    '.-', label=labels[i])
            plt.xlabel('Timesteps')
            plt.ylabel('Error')
        plt.legend(loc=2)
    plt.axis([0, 150, -5, 80])
    plt.show()

if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d','movie2d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, nargs='*',# default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default='none',
                   help='Name of tracer file to display underneath particle trajectories')
    args = p.parse_args()

    particleplotting(args.particlefile, args.tracerfile, mode=args.mode)
