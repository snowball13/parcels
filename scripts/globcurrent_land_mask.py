#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from parcels import Grid


tracerfile = 'examples/GlobCurrent_example_data/20020101000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc'
tracerlon = 'lon'
tracerlat = 'lat'
tracerfield = 'northward_eulerian_current_velocity'

tfile = Dataset(tracerfile, 'r')     # Load NETCDF4 file.
X = tfile.variables[tracerlon]
Y = tfile.variables[tracerlat]
P = tfile.variables[tracerfield]


P = P[0, :]                          # Drop redundant first dimension - only one data file anyway.
land_mask = np.zeros(np.shape(P))


for i in xrange(P.shape[0]):         # Rewrite land_mask to ouline the coast.
    for j in xrange(P.shape[1]):
        if np.isnan(P[i, j]) is False:
            land_mask[i, j] = 1
        else:
            land_mask[i, j] = 0       # 1 represents land, 0 represents ocean


plt.contourf(range(P.shape[1]), range(P.shape[0]), land_mask)
plt.show()


rootgrp = Dataset("test.nc", "w",
                  format="NETCDF4")   # Write to NETCDF4 file test.nc


lats = X[:, ]
lons = Y[:, ]

lat = rootgrp.createDimension('lat', len(lats))
lon = rootgrp.createDimension('lon', len(lons))
time = rootgrp.createDimension('time', None)

latitudes = rootgrp.createVariable('latitude', 'f8', ('lat', ))
longitudes = rootgrp.createVariable('longitude', 'f8', ('lon', ))
landmass = rootgrp.createVariable('land', 'f8', ('lat', 'lon'))
times = rootgrp.createVariable("time", "f8", ("time",))

latitudes[:] = lats
longitudes[:] = lons
landmass[:, :] = land_mask

rootgrp.close()


filenames = {'landmass': 'test.nc'}
variables = {'landmass': 'land'}
dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}

# Attempt to run the file but it fails - the landmass doesn't have a time dimension.
grid = Grid.from_netcdf(filenames, variables, dimensions)
