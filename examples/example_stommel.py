from parcels import ScipyParticle, JITParticle, Variable
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from argparse import ArgumentParser
import numpy as np
import pytest
from datetime import timedelta as delta
from scripts.allgrids import import_grid


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def UpdateP(particle, grid, time, dt):
    particle.p = grid.P[time, particle.lon, particle.lat]


def stommel_example(npart=1, mode='jit', verbose=False, method=AdvectionRK4):

    grid = import_grid().stommel_grid()
    filename = 'stommel'
    grid.write(filename)

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class MyParticle(ParticleClass):
        p = Variable('p', dtype=np.float32, initial=0.)
        p_start = Variable('p_start', dtype=np.float32, initial=0.)

    pset = grid.ParticleSet(size=npart, pclass=MyParticle,
                            start=(100, 5000), finish=(200, 5000))
    for particle in pset:
        particle.p_start = grid.P[0., particle.lon, particle.lat]

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execute for 50 days, with 5min timesteps and hourly output
    runtime = delta(days=50)
    dt = delta(minutes=5)
    interval = delta(hours=12)
    print("Stommel: Advecting %d particles for %s" % (npart, runtime))
    pset.execute(method + pset.Kernel(UpdateP), runtime=runtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="StommelParticle"), show_movie=False)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_stommel_grid(mode):
    psetRK4 = stommel_example(1, mode=mode, method=method['RK4'])
    psetRK45 = stommel_example(1, mode=mode, method=method['RK45'])
    assert np.allclose([p.lon for p in psetRK4], [p.lon for p in psetRK45], rtol=1e-3)
    assert np.allclose([p.lat for p in psetRK4], [p.lat for p in psetRK45], rtol=1e-3)
    err_adv = np.array([abs(p.p_start - p.p) for p in psetRK4])
    assert(err_adv <= 1.e-1).all()
    err_smpl = np.array([abs(p.p - psetRK4.grid.P[0., p.lon, p.lat]) for p in psetRK4])
    assert(err_smpl <= 1.e-1).all()


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection in the steady-state solution of the Stommel equation""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    p.add_argument('-p', '--particles', type=int, default=1,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('-m', '--method', choices=('RK4', 'EE', 'RK45'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()

    stommel_example(args.particles, mode=args.mode,
                    verbose=args.verbose, method=method[args.method])
