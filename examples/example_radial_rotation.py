from parcels import ScipyParticle, JITParticle
from parcels import AdvectionRK4
from scripts.allgrids import import_grid
import numpy as np
from datetime import timedelta as delta
import math
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def true_values(age):  # Calculate the expected values for particle 2 at the endtime.

    x = 20*math.sin(2*np.pi*age/(24.*60.**2)) + 30.
    y = 20*math.cos(2*np.pi*age/(24.*60.**2)) + 30.

    return [x, y]


def rotation_example(grid, mode='jit', method=AdvectionRK4):

    npart = 2          # Test two particles on the rotating grid.
    pset = grid.ParticleSet(size=npart, pclass=ptype[mode],
                            start=(30., 30.),
                            finish=(30., 50.))  # One particle in centre, one on periphery of grid.

    endtime = delta(hours=17)
    dt = delta(minutes=5)
    interval = delta(hours=1)

    pset.execute(method, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="RadialParticle"), show_movie=False)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_rotation_example(mode):
    grid = import_grid().radial_rotation_grid()
    pset = rotation_example(grid, mode=mode)
    assert(pset[0].lon == 30. and pset[0].lat == 30.)  # Particle at centre of grid remains stationary.
    vals = true_values(pset[1].time)
    assert(np.allclose(pset[1].lon, vals[0], 1e-5))    # Check advected values against calculated values.
    assert(np.allclose(pset[1].lat, vals[1], 1e-5))

if __name__ == "__main__":
    filename = 'radial_rotation'
    grid = import_grid().radial_rotation_grid()
    grid.write(filename)

    rotation_example(grid)
