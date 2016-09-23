from parcels import Grid
import math
import numpy as np
from datetime import timedelta as delta


def moving_eddies_grid(xdim=200, ydim=350):
    """Generate a grid encapsulating the flow field consisting of two
    moving eddies, one moving westward and the other moving northwestward.

    Note that this is not a proper geophysical flow. Rather, a Gaussian eddy is moved
    artificially with uniform velocities. Velocities are calculated from geostrophy.
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 25. * 86400., 86400., dtype=np.float64)

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, 4, xdim, dtype=np.float32)
    lat = np.linspace(45, 52, ydim, dtype=np.float32)

    # Grid spacing in m
    def cosd(x):
        return math.cos(math.radians(float(x)))
    dx = (lon[1] - lon[0]) * 1852 * 60 * cosd(lat.mean())
    dy = (lat[1] - lat[0]) * 1852 * 60

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Some constants
    corio_0 = 1.e-4  # Coriolis parameter
    h0 = 1  # Max eddy height
    sig = 0.5  # Eddy e-folding decay scale (in degrees)
    g = 10  # Gravitational constant
    eddyspeed = 0.1  # Translational speed in m/s
    dX = eddyspeed * 86400 / dx  # Grid cell movement of eddy max each day
    dY = eddyspeed * 86400 / dy  # Grid cell movement of eddy max each day

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
        hymax_1 = lat.size / 7.
        hxmax_1 = .75 * lon.size - dX * t
        hymax_2 = 3. * lat.size / 7. + dY * t
        hxmax_2 = .75 * lon.size - dX * t

        P[:, :, t] = h0 * np.exp(-(x-hxmax_1)**2/(sig*lon.size/4.)**2-(y-hymax_1)**2/(sig*lat.size/7.)**2)
        P[:, :, t] += h0 * np.exp(-(x-hxmax_2)**2/(sig*lon.size/4.)**2-(y-hymax_2)**2/(sig*lat.size/7.)**2)

        V[:-1, :, t] = -np.diff(P[:, :, t], axis=0) / dx / corio_0 * g
        V[-1, :, t] = V[-2, :, t]  # Fill in the last column

        U[:, :-1, t] = np.diff(P[:, :, t], axis=1) / dy / corio_0 * g
        U[:, -1, t] = U[:, -2, t]  # Fill in the last row

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, field_data={'P': P})


def stommel_grid(xdim=200, ydim=200):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.linspace(0., 100000. * 86400., 2, dtype=np.float64)

    # Some constants
    A = 100
    eps = 0.05
    a = 10000
    b = 10000

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    [x, y] = np.mgrid[:lon.size, :lat.size]
    l1 = (-1 + math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
    l2 = (-1 - math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
    c1 = (1 - math.exp(l2)) / (math.exp(l2) - math.exp(l1))
    c2 = -(1 + c1)
    for t in range(time.size):
        for i in range(lon.size):
            for j in range(lat.size):
                xi = lon[i] / a
                yi = lat[j] / b
                P[i, j, t] = A * (c1*math.exp(l1*xi) + c2*math.exp(l2*xi) + 1) * math.sin(math.pi * yi)
        for i in range(lon.size-2):
            for j in range(lat.size):
                V[i+1, j, t] = (P[i+2, j, t] - P[i, j, t]) / (2 * a / xdim)
        for i in range(lon.size):
            for j in range(lat.size-2):
                U[i, j+1, t] = -(P[i, j+2, t] - P[i, j, t]) / (2 * b / ydim)

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, field_data={'P': P}, mesh='flat')


def peninsula_grid(xdim, ydim):
    """Construct a grid encapsulating the flow field around an
    idealised peninsula.

    :param xdim: Horizontal dimension of the generated grid
    :param xdim: Vertical dimension of the generated grid

    The original test description can be found in Fig. 2.2.3 in:
    North, E. W., Gallego, A., Petitgas, P. (Eds). 2009. Manual of
    recommended practices for modelling physical - biological
    interactions during fish early life.
    ICES Cooperative Research Report No. 295. 111 pp.
    http://archimer.ifremer.fr/doc/00157/26792/24888.pdf

    Note that the problem is defined on an A-grid while NEMO
    normally returns C-grids. However, to avoid accuracy
    problems with interpolation from A-grid to C-grid, we
    return NetCDF files that are on an A-grid.
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)

    # Generate the original test setup on A-grid in km
    dx = 100. / xdim / 2.
    dy = 50. / ydim / 2.
    La = np.linspace(dx, 100.-dx, xdim, dtype=np.float32)
    Wa = np.linspace(dy, 50.-dy, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((xdim, ydim), dtype=np.float32)
    V = np.zeros((xdim, ydim), dtype=np.float32)
    W = np.zeros((xdim, ydim), dtype=np.float32)
    P = np.zeros((xdim, ydim), dtype=np.float32)

    u0 = 1
    x0 = 50.
    R = 0.32 * 50.

    # Create the fields
    x, y = np.meshgrid(La, Wa, sparse=True, indexing='ij')
    P = u0*R**2*y/((x-x0)**2+y**2)-u0*y
    U = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
    V = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

    # Set land points to NaN
    I = P >= 0.
    U[I] = np.nan
    V[I] = np.nan
    W[I] = np.nan

    # Convert from km to lat/lon
    lon = La / 1.852 / 60.
    lat = Wa / 1.852 / 60.

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, field_data={'P': P})


def radial_rotation_grid(xdim=200, ydim=200):  # Define 2D flat, square grid for testing purposes.

    lon = np.linspace(0, 60, xdim, dtype=np.float32)
    lat = np.linspace(0, 60, ydim, dtype=np.float32)

    x0 = 30.                                   # Define the origin to be the centre of the grid.
    y0 = 30.

    U = np.zeros((xdim, ydim), dtype=np.float32)
    V = np.zeros((xdim, ydim), dtype=np.float32)

    T = delta(days=1)
    omega = 2*np.pi/T.total_seconds()          # Define the rotational period as 1 day.

    for i in range(lon.size):
        for j in range(lat.size):

            r = np.sqrt((lon[i]-x0)**2 + (lat[j]-y0)**2)  # Define radial displacement.
            assert(r >= 0.)
            assert(r <= np.sqrt(x0**2 + y0**2))

            theta = math.atan2((lat[j]-y0), (lon[i]-x0))  # Define the polar angle.
            assert(abs(theta) <= np.pi)

            U[i, j] = r * math.sin(theta) * omega
            V[i, j] = -r * math.cos(theta) * omega

    return Grid.from_data(U, lon, lat, V, lon, lat, mesh='flat')


# Define some constants.
u_g = .04  # Geostrophic current
u_0 = .3  # Initial speed in x dirrection. v_0 = 0
gamma = 1./delta(days=2.89).total_seconds()  # Dissipitave effects due to viscousity.
gamma_g = 1./delta(days=28.9).total_seconds()
f = 1.e-4  # Coriolis parameter.
start_lon = [10000.]  # Define the start longitude and latitude for the particle.
start_lat = [10000.]


def decaying_moving_eddy_grid(xdim=2, ydim=2):  # Define 2D flat, square grid for testing purposes.
    """Simulate an ocean that accelerates subject to Coriolis force
    and dissipative effects, upon which a geostrophic current is
    superimposed.

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 2. * 86400., 60.*5., dtype=np.float64)
    lon = np.linspace(0, 20000, xdim, dtype=np.float32)
    lat = np.linspace(5000, 12000, ydim, dtype=np.float32)

    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    for t in range(time.size):
        U[:, :, t] = u_g*np.exp(-gamma_g*time[t]) + (u_0-u_g)*np.exp(-gamma*time[t])*np.cos(f*time[t])
        V[:, :, t] = -(u_0-u_g)*np.exp(-gamma*time[t])*np.sin(f*time[t])

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, mesh='flat')
