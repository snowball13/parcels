#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
import matplotlib.animation as animation
import math

def ground_truth(lon,lat):
    #May not be exactly fitted to the curve because grids may be different
    day = 11.6
    r = 1 / (day * 86400)
    beta = 4e-9
    a = 10000
    e_s = r / (beta * a)
    psi = (1 - np.exp(-lon * math.pi / 180 / e_s) - lon *\
        math.pi / 180) * math.pi * np.sin(math.pi ** 2 * lat / 180)
    return psi

def particleplotting(filenames,tracerfile, mode):
    """Quick and simple plotting of PARCELS trajectories"""

    filename = filenames[0]
    pfile=Dataset(filename,'r')
    lon = np.empty((np.size(pfile.variables['lon'],0),np.size(pfile.variables['lon'],1),len(filenames)))
    lat = np.empty((np.size(pfile.variables['lat'],0),np.size(pfile.variables['lat'],1),len(filenames)))
    z = np.empty((np.size(pfile.variables['z'],0),np.size(pfile.variables['z'],1),len(filenames)))
    time = pfile.variables['time'][:,:]
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(3, 2)
    labels = ('Analytical', 'RK4', 'EE')    #If you call them in this order. Could add option to set labels
    for i in range(len(filenames)):
        filename = filenames[i]
        pfile=Dataset(filename,'r')
        lon[:,:,i] = pfile.variables['lon'][:,:]
        lat[:,:,i] = pfile.variables['lat'][:,:]
        z[:,:,i] = pfile.variables['z'][:,:]
        x_grid = np.linspace(0, 60, 1000)
        y_grid = np.linspace(0, 60, 1000)
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        psi_grid = ground_truth(x_grid, y_grid)
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
            ax1 = fig.add_subplot(gs[:,0])
            levels = ground_truth(lon[:,0,i], lat[:,0,i])
            ax1.contour(x_grid, y_grid, psi_grid, levels[:], colors='0.4', linestyles='dashed')
            ax1.plot(np.transpose(lon[:,:,i]),np.transpose(lat[:,:,i]),'.-')
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

    #Plot error
    #fig2, ax2 = plt.subplots()
    ax2 = fig.add_subplot(gs[0,1])
    for i in range(len(filenames)):
        psi = ground_truth(lon[:,:,i],lat[:,:,i])
        for j in range(len(psi)):
            ax2.plot(psi[j,:] - psi[j,0])
    plt.title("Errors")
    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(np.transpose(np.diff(time)))
    plt.title("Step size")
    ax4 = fig.add_subplot(gs[2,1])
#    print np.shape(np.transpose(np.squeeze(lon)))
    ax4.plot(np.transpose(np.squeeze(lon[:,:])))
    plt.title("Longitude")
    plt.show()

if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d','movie2d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, nargs='*',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default='none',
                   help='Name of tracer file to display underneath particle trajectories')
    args = p.parse_args()

    particleplotting(args.particlefile, args.tracerfile, mode=args.mode)
