from parcels import Grid
import math
import numpy as np
from datetime import timedelta as delta


class import_grid:

    def __init__(self, xdim=300, ydim=300, days=None, timestep=None):
        self.xdim = xdim
        self.ydim = ydim
        # Define NEMO grid variables. Generate time dimension for time-indep. grids.
        self.depth = np.zeros(1, dtype=np.float32)
        if days is None:
            self.time = np.zeros(1, dtype=np.float64)
        else:
            self.time = np.arange(0., days * 86400., timestep)
        # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
        # surface height) all on A-grid
        self.U = np.zeros((self.xdim, self.ydim, self.time.size), dtype=np.float32)
        self.V = np.zeros((self.xdim, self.ydim, self.time.size), dtype=np.float32)
        self.W = np.zeros((self.xdim, self.ydim, self.time.size), dtype=np.float32)
        self.P = np.zeros((self.xdim, self.ydim, self.time.size), dtype=np.float32)
        self.corio_0 = 1.e-4  # Coriolis parameter

    def moving_eddies_grid(self):
        """Generate a grid encapsulating the flow field consisting of two
        moving eddies, one moving westward and the other moving northwestward.

        Note that this is not a proper geophysical flow. Rather, a Gaussian eddy is moved
        artificially with uniform velocities. Velocities are calculated from geostrophy.
        """
        time = np.arange(0., 25. * 86400., 86400., dtype=np.float64)
        self.U = np.zeros((self.xdim, self.ydim, time.size), dtype=np.float32)  # Need to redefine fields for new time dimension.
        self.V = np.zeros((self.xdim, self.ydim, time.size), dtype=np.float32)
        self.P = np.zeros((self.xdim, self.ydim, time.size), dtype=np.float32)

        # Coordinates of the test grid (on A-grid in deg)
        lon = np.linspace(0, 4, self.xdim, dtype=np.float32)
        lat = np.linspace(45, 52, self.ydim, dtype=np.float32)

        # Grid spacing in m
        def cosd(x):
            return math.cos(math.radians(float(x)))
        dx = (lon[1] - lon[0]) * 1852 * 60 * cosd(lat.mean())
        dy = (lat[1] - lat[0]) * 1852 * 60

        # Some constants
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

            self.P[:, :, t] = h0 * np.exp(-(x-hxmax_1)**2/(sig*lon.size/4.)**2-(y-hymax_1)**2/(sig*lat.size/7.)**2)
            self.P[:, :, t] += h0 * np.exp(-(x-hxmax_2)**2/(sig*lon.size/4.)**2-(y-hymax_2)**2/(sig*lat.size/7.)**2)

            self.V[:-1, :, t] = -np.diff(self.P[:, :, t], axis=0) / dx / self.corio_0 * g
            self.V[-1, :, t] = self.V[-2, :, t]  # Fill in the last column

            self.U[:, :-1, t] = np.diff(self.P[:, :, t], axis=1) / dy / self.corio_0 * g
            self.U[:, -1, t] = self.U[:, -2, t]  # Fill in the last row

        return Grid.from_data(self.U, lon, lat, self.V, lon, lat, self.depth, self.time, field_data={'P': self.P})

    def stommel_grid(self):
        """Simulate a periodic current along a western boundary, with significantly
        larger velocities along the western edge than the rest of the region

        The original test description can be found in: N. Fabbroni, 2009,
        Numerical Simulation of Passive tracers dispersion in the sea,
        Ph.D. dissertation, University of Bologna
        http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
        """
        # Some constants
        A = 100
        eps = 0.05
        a = 10000
        b = 10000

        # Coordinates of the test grid (on A-grid in deg)
        lon = np.linspace(0, a, self.xdim, dtype=np.float32)
        lat = np.linspace(0, b, self.ydim, dtype=np.float32)

        [x, y] = np.mgrid[:lon.size, :lat.size]
        l1 = (-1 + math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
        l2 = (-1 - math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
        c1 = (1 - math.exp(l2)) / (math.exp(l2) - math.exp(l1))
        c2 = -(1 + c1)
        for t in range(self.time.size):
            for i in range(lon.size):
                for j in range(lat.size):
                    xi = lon[i] / a
                    yi = lat[j] / b
                    self.P[i, j, t] = A * (c1*math.exp(l1*xi) + c2*math.exp(l2*xi) + 1) * math.sin(math.pi * yi)
            for i in range(lon.size-2):
                for j in range(lat.size):
                    self.V[i+1, j, t] = (self.P[i+2, j, t] - self.P[i, j, t]) / (2 * a / self.xdim)
            for i in range(lon.size):
                for j in range(lat.size-2):
                    self.U[i, j+1, t] = -(self.P[i, j+2, t] - self.P[i, j, t]) / (2 * b / self.ydim)

        return Grid.from_data(self.U, lon, lat, self.V, lon, lat, self.depth, self.time, field_data={'P': self.P}, mesh='flat')

    def peninsula_grid(self):
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
        # Generate the original test setup on A-grid in km
        dx = 100. / self.xdim / 2.
        dy = 50. / self.ydim / 2.
        La = np.linspace(dx, 100.-dx, self.xdim, dtype=np.float32)
        Wa = np.linspace(dy, 50.-dy, self.ydim, dtype=np.float32)

        u0 = 1
        x0 = 50.
        R = 0.32 * 50.

        # Create the fields
        x, y = np.meshgrid(La, Wa, sparse=True, indexing='ij')
        self.P = u0*R**2*y/((x-x0)**2+y**2)-u0*y
        self.U = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
        self.V = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

        # Set land points to NaN
        I = self.P >= 0.
        self.U[I] = np.nan
        self.V[I] = np.nan
        self.W[I] = np.nan

        # Convert from km to lat/lon
        lon = La / 1.852 / 60.
        lat = Wa / 1.852 / 60.

        return Grid.from_data(self.U, lon, lat, self.V, lon, lat, self.depth, self.time, field_data={'P': self.P})

    def radial_rotation_grid(self):  # Define 2D flat, square grid for testing purposes.

        lon = np.linspace(0, 60, self.xdim, dtype=np.float32)
        lat = np.linspace(0, 60, self.ydim, dtype=np.float32)

        x0 = 30.                                   # Define the origin to be the centre of the grid.
        y0 = 30.

        omega = 2*np.pi/delta(days=1).total_seconds()          # Define the rotational period as 1 day.

        for i in range(lon.size):
            for j in range(lat.size):

                r = np.sqrt((lon[i]-x0)**2 + (lat[j]-y0)**2)  # Define radial displacement.
                assert(r >= 0.)
                assert(r <= np.sqrt(x0**2 + y0**2))

                theta = math.atan2((lat[j]-y0), (lon[i]-x0))  # Define the polar angle.
                assert(abs(theta) <= np.pi)

                self.U[i, j] = r * math.sin(theta) * omega
                self.V[i, j] = -r * math.cos(theta) * omega

        return Grid.from_data(self.U, lon, lat, self.V, lon, lat, mesh='flat')

    def decaying_moving_eddy_grid(self):  # Define 2D flat, square grid for testing purposes.
        """Simulate an ocean that accelerates subject to Coriolis force
        and dissipative effects, upon which a geostrophic current is
        superimposed.

        The original test description can be found in: N. Fabbroni, 2009,
        Numerical Simulation of Passive tracers dispersion in the sea,
        Ph.D. dissertation, University of Bologna
        http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
        """
        # Define some constants.
        u_g = .04  # Geostrophic current
        u_0 = .3  # Initial speed in x dirrection. v_0 = 0
        gamma = 1./delta(days=2.89).total_seconds()  # Dissipitave effects due to viscousity.
        gamma_g = 1./delta(days=28.9).total_seconds()

        lon = np.linspace(0, 20000, 2, dtype=np.float32)  # No spatial dependence. Use 2x2 grid.
        lat = np.linspace(5000, 12000, 2, dtype=np.float32)

        for t in range(self.time.size):
            self.U[:, :, t] = u_g*np.exp(-gamma_g*self.time[t]) + (u_0-u_g)*np.exp(-gamma*self.time[t])*np.cos(self.corio_0*self.time[t])
            self.V[:, :, t] = -(u_0-u_g)*np.exp(-gamma*self.time[t])*np.sin(self.corio_0*self.time[t])

        return Grid.from_data(self.U, lon, lat, self.V, lon, lat, self.depth, self.time, mesh='flat')
