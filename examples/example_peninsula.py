from parcels import Grid, ScipyParticle, JITParticle, Variable
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from argparse import ArgumentParser
import numpy as np
import math  # NOQA
import pytest
from datetime import timedelta as delta
from scripts.allgrids import peninsula_grid


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def UpdateP(particle, grid, time, dt):
    particle.p = grid.P[time, particle.lon, particle.lat]


def pensinsula_example(grid, npart, mode='jit', degree=1,
                       verbose=False, output=True, method=AdvectionRK4):
    """Example configuration of particle flow around an idealised Peninsula

    :arg filename: Basename of the input grid file set
    :arg npart: Number of particles to intialise"""

    # First, we define a custom Particle class to which we add a
    # custom variable, the initial stream function value p.
    # We determine the particle base class according to mode.
    class MyParticle(ptype[mode]):
        # JIT compilation requires a-priori knowledge of the particle
        # data structure, so we define additional variables here.
        p = Variable('p', dtype=np.float32, initial=0.)
        p_start = Variable('p_start', dtype=np.float32, initial=0.)

        def __repr__(self):
            """Custom print function which overrides the built-in"""
            return "P(%.4f, %.4f)[p=%.5f, p_start=%f]" % (self.lon, self.lat,
                                                          self.p, self.p_start)

    # Initialise particles
    x = 3. * (1. / 1.852 / 60)  # 3 km offset from boundary
    y = (grid.U.lat[0] + x, grid.U.lat[-1] - x)  # latitude range, including offsets
    pset = grid.ParticleSet(npart, pclass=MyParticle, start=(x, y[0]), finish=(x, y[1]))
    for particle in pset:
        particle.p_start = grid.P[0., particle.lon, particle.lat]

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Advect the particles for 24h
    time = delta(hours=24)
    dt = delta(minutes=5)
    k_adv = pset.Kernel(method)
    k_p = pset.Kernel(UpdateP)
    out = pset.ParticleFile(name="MyParticle") if output else None
    interval = delta(hours=1) if output else -1
    print("Peninsula: Advecting %d particles for %s" % (npart, str(time)))
    pset.execute(k_adv + k_p, endtime=time, dt=dt, output_file=out, interval=interval)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_peninsula_grid(mode):
    """Execute peninsula test from grid generated in memory"""
    grid = peninsula_grid(100, 50)
    pset = pensinsula_example(grid, 100, mode=mode, degree=1)
    # Test advection accuracy by comparing streamline values
    err_adv = np.array([abs(p.p_start - p.p) for p in pset])
    assert(err_adv <= 1.e-3).all()
    # Test grid sampling accuracy by comparing kernel against grid sampling
    err_smpl = np.array([abs(p.p - pset.grid.P[0., p.lon, p.lat]) for p in pset])
    assert(err_smpl <= 1.e-3).all()


@pytest.fixture(scope='module')
def gridfile():
    """Generate grid files for peninsula test"""
    filename = 'peninsula'
    grid = peninsula_grid(100, 50)
    grid.write(filename)
    return filename


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_peninsula_file(gridfile, mode):
    """Open grid files and execute"""
    grid = Grid.from_nemo(gridfile, extra_vars={'P': 'P'})
    pset = pensinsula_example(grid, 100, mode=mode, degree=1)
    # Test advection accuracy by comparing streamline values
    err_adv = np.array([abs(p.p_start - p.p) for p in pset])
    assert(err_adv <= 1.e-3).all()
    # Test grid sampling accuracy by comparing kernel against grid sampling
    err_smpl = np.array([abs(p.p - pset.grid.P[0., p.lon, p.lat]) for p in pset])
    assert(err_smpl <= 1.e-3).all()


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=20,
                   help='Number of particles to advect')
    p.add_argument('-d', '--degree', type=int, default=1,
                   help='Degree of spatial interpolation')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('-o', '--nooutput', action='store_true', default=False,
                   help='Suppress trajectory output')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, nargs=2, default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-m', '--method', choices=('RK4', 'EE', 'RK45'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()

    if args.grid is not None:
        filename = 'peninsula'
        grid = peninsula_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid file set
    grid = Grid.from_nemo('peninsula', extra_vars={'P': 'P'})

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("pensinsula_example(grid, args.particles, mode=args.mode,\
                                   degree=args.degree, verbose=args.verbose,\
                                   output=not args.nooutput, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        pensinsula_example(grid, args.particles, mode=args.mode,
                           degree=args.degree, verbose=args.verbose,
                           output=not args.nooutput, method=method[args.method])
