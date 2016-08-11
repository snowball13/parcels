from netCDF4 import Dataset
from parcels import Grid, Particle
from example_globcurrent import set_globcurrent_grid
from datetime import timedelta as delta
import numpy as np


def set_land_mask_grid():

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
                land_mask[i, j] = 0      # 1 represents land, 0 represents ocean

    grid.land_mask = land_mask  # NEED TO MAKE A PROPER PARCELS FIELD

    return grid


def land_kernel(particle, grid, time, dt):

    if grid.land_mask[particle.lon, particle.lat] == 1:
        particle.on_land = True
    else:
        particle.on_land = False


def test_land_mask_grid(grid, npart=2):

    class MyParticle(Particle):
        user_vars = {'on_land': bool}

        def __init__(self, *args, **kwargs):
            super(MyParticle, self).__init__(*args, **kwargs)

    x = 24.875
    y = np.array([-39.125, -31.125])
    pset = grid.ParticleSet(npart, pclass=MyParticle, start=(x, y[0]), finish=(x, y[1]))

    runtime = delta(days=1)
    dt = delta(minutes=5)
    interval = delta(hours=2)
    k_mask = pset.Kernel(land_kernel)

    pset.execute(k_mask, runtime=runtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="maskParticle"),
                 show_movie = False)

    assert(pset[0].on_land == False)
    assert(pset[1].on_land == True)


if __name__ == "__main__":

    grid = set_land_mask_grid()
    test_land_mask_grid(grid, npart=2)
