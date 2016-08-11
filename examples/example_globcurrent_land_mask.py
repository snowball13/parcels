#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
from parcels import Grid


def test_globcurrent_land_mask():
    grid = set_globcurrent_grid()

    tracerfile = 'examples/GlobCurrent_example_data/20020101000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc'

    tfile = Dataset(tracerfile, 'r')     # Load NETCDF4 file.
    X = tfile.variables['lon']
    Y = tfile.variables['lat']
    P = tfile.variables['northward_eulerian_current_velocity']

    P = P[0, :]                          # Drop redundant first dimension - only one data file anyway.
    land_mask = np.zeros(np.shape(P))

    for i in xrange(P.shape[0]):         # Rewrite land_mask to ouline the coast.
        for j in xrange(P.shape[1]):
            if not np.isnan(P[i, j]):
                land_mask[i, j] = 1
            else:
                land_mask[i, j] = 0       # 1 represents land, 0 represents ocean

    grid.land_mask = land_mask
