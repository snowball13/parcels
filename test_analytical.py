from parcels import NEMOGrid, Particle, JITParticle, AdvectionRK4, AdvectionEE
from argparse import ArgumentParser
import numpy as np
import math
import pytest

def ground_truth(x_0, y_0, t):
    f = 0.0001  #Coriolis frequency
    u_0 = 0.3   
    u_g = 0.04
    gamma = 1 / (86400 * 2.89)
    gamma_g = 1 / (86400 * 28.9)
    x = x_0 + u_g / gamma_g * (1 - np.exp(-gamma_g * t)) + (u_0 - u_g) * f /\
        (f ** 2 + gamma ** 2) * (gamma / f + np.exp(-gamma * t) *\
        (math.sin(f * t) - gamma / f * math.cos(f * t)))
    y = y_0 - (u_0 - u_g) * f / (f ** 2 + gamma ** 2) *\
        (1 - np.exp(-gamma * t) * (np.cos(f * t) + gamma / f * np.sin(f * t)))


def analytical_eddies_grid(xdim=200, ydim=200):
    """Generate a grid encapsulating the flow field consisting of two
    moving eddies, one moving westward and the other moving northwestward.

    The original test description can be found in: K. Doos,
    J. Kjellsson and B. F. Jonsson. 2013 TRACMASS - A Lagrangian
    Trajectory Model, in Preventive Methods for Coastal Protection,
    T. Soomere and E. Quak (Eds.),
    http://www.springer.com/gb/book/9783319004396
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 25. * 86400., 864., dtype=np.float64)

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, 4, xdim, dtype=np.float32)
    lat = np.linspace(40, 50, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Some constants
    f = 1.e-4  # Coriolis parameter
    u_0 = 0.3
    u_g = 0.04
    gamma = 1/(86400. * 2.89)
    gamma_g = 1/(86400. * 28.9)

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
#        U[:,:,t] = u_0 * math.cos(f * time[t])
#        V[:,:,t] = -u_0 * math.sin(f * time[t])
        U[:, :, t] = u_g * np.exp(-gamma_g * time[t]) + (u_0 - u_g) *\
                    np.exp(-gamma * time[t]) * math.cos(f * time[t])
        V[:, :, t] = -(u_0 * u_g) * np.exp(-gamma * time[t]) * math.sin(f * time[t])
        #print f * time[t]

    return NEMOGrid.from_data(U, lon, lat, V, lon, lat,
                              depth, time, field_data={'P': P})


def analytical_eddies_example(grid, npart=1, mode='jit', verbose=False,
                          method=AdvectionRK4):
    """Configuration of a particle set that follows two moving eddies

    :arg grid: :class NEMOGrid: that defines the flow field
    :arg npart: Number of particles to intialise"""

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle

    pset = grid.ParticleSet(size=npart, pclass=ParticleClass,
                            start=(2., 49.), finish=(2., 49.))

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execte for 25 days, with 5min timesteps and hourly output
    hours = 10*24   #25*24
    substeps = 6
    print("MovingEddies: Advecting %d particles for %d timesteps"
          % (npart, hours * substeps))
    pset.execute(method, timesteps=hours*substeps, dt = 300.,#dt=300.,
                 output_file=pset.ParticleFile(name="AnalyticalParticle"),
                 output_steps=substeps)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_analytic_eddies_grid(mode):
    grid = analytical_eddies_grid()
    pset = analytical_eddies_example(grid, 1, mode=mode)
    assert(pset[0].lon < 0.5 and 45.8 < pset[0].lat < 46.15)
    assert(pset[1].lon < 0.5 and 50.4 < pset[1].lat < 50.7)


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    p.add_argument('-p', '--particles', type=int, default=1,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, nargs=2, default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-m', '--method', choices=('RK4', 'EE'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()
    filename = 'analytical_eddies'
    
    method = locals()['Advection' + args.method]

    # Generate grid files according to given dimensions
    if args.grid is not None:
        grid = analytical_eddies_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid files
    grid = NEMOGrid.from_file(filename)

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("analytical_eddies_example(grid, args.particles, mode=args.mode, \
                              verbose=args.verbose)",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        analytical_eddies_example(grid, args.particles, mode=args.mode,
                              verbose=args.verbose, method=method)
