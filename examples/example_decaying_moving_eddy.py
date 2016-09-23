from parcels import Grid, ScipyParticle, JITParticle
from parcels import AdvectionRK4
import numpy as np
from scripts.allgrids import decaying_moving_eddy_grid
from datetime import timedelta as delta
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}

# Define some constants.
u_g = .04  # Geostrophic current
u_0 = .3  # Initial speed in x dirrection. v_0 = 0
gamma = 1./delta(days=2.89).total_seconds()  # Dissipitave effects due to viscousity.
gamma_g = 1./delta(days=28.9).total_seconds()
f = 1.e-4  # Coriolis parameter.
start_lon = [10000.]  # Define the start longitude and latitude for the particle.
start_lat = [10000.]


def true_values(t, x_0, y_0):  # Calculate the expected values for particles at the endtime, given their start location.
    x = x_0 + (u_g/gamma_g)*(1-np.exp(-gamma_g*t)) + f*((u_0-u_g)/(f**2 + gamma**2))*((gamma/f) + np.exp(-gamma*t)*(np.sin(f*t) - (gamma/f)*np.cos(f*t)))
    y = y_0 - ((u_0-u_g)/(f**2+gamma**2))*f*(1 - np.exp(-gamma*t)*(np.cos(f*t) + (gamma/f)*np.sin(f*t)))

    return np.array([x, y])


def decaying_moving_example(grid, mode='scipy', method=AdvectionRK4):
    pset = grid.ParticleSet(size=1, pclass=ptype[mode], lon=start_lon, lat=start_lat)

    endtime = delta(days=2)
    dt = delta(minutes=5)
    interval = delta(hours=1)

    pset.execute(method, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="DecayingMovingParticle"), show_movie=False)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_rotation_example(mode):
    grid = decaying_moving_eddy_grid()
    pset = decaying_moving_example(grid, mode=mode)
    vals = true_values(pset[0].time, start_lon, start_lat)  # Calculate values for the particle.
    assert(np.allclose(np.array([[pset[0].lon], [pset[0].lat]]), vals, 1e-2))   # Check advected values against calculated values.


if __name__ == "__main__":
    filename = 'decaying_moving_eddy'
    grid = decaying_moving_eddy_grid()
    grid.write(filename)

    pset = decaying_moving_example(grid)
