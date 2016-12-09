import numpy as np
import netCDF4


__all__ = ['ParticleFile']


class ParticleFile(object):

    def __init__(self, name, particleset, type='array'):
        """Initialise netCDF4.Dataset for trajectory output.

        The output follows the format outlined in the Discrete
        Sampling Geometries section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        Both 'Orthogonal multidimensional array' and 'Indexed ragged array' represenation
        are implemented. The former is simpler to post-process, but the latter is required
        when particles will be added and deleted during the .execute (i.e. the number of
        particles in the pset is not fixed).

        Developer note: We cannot use xray.Dataset here, since it does
        not yet allow incremental writes to disk:
        https://github.com/xray/xray/issues/199

        :param name: Basename of the output file
        :param particlset: ParticleSet to output
        :param user_vars: A list of additional user defined particle variables to write
        :param type: Either 'array' for default matrix style, or 'indexed' for indexed ragged array
        """
        self.type = type
        self.dataset = netCDF4.Dataset("%s.nc" % name, "w", format="NETCDF4")
        self.dataset.createDimension("obs", None)
        if self.type is 'array':
            self.dataset.createDimension("trajectory", particleset.size)
            coords = ("trajectory", "obs")
        elif self.type is 'indexed':
            coords = ("obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"

        # Create ID variable according to CF conventions
        if self.type is 'array':
            self.trajectory = self.dataset.createVariable("trajectory", "i4", ("trajectory",))
            self.trajectory.long_name = "Unique identifier for each particle"
            self.trajectory.cf_role = "trajectory_id"
            self.trajectory[:] = np.array([p.trajectory for p in particleset])
        elif self.type is 'indexed':
            self.trajectory = self.dataset.createVariable("trajectory", "i4", ("obs",))
            self.trajectory.long_name = "index of trajectory this obs belongs to"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if particleset.time_origin == 0:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(particleset.time_origin)
            self.time.calendar = "julian"
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", coords, fill_value=np.nan)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", coords, fill_value=np.nan)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", coords, fill_value=np.nan)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        self.user_vars = []
        for v in particleset.ptype.variables:
            if v.name in ['time', 'lat', 'lon', 'z', 'trajectory']:
                continue
            if v.to_write is True:
                setattr(self, v.name, self.dataset.createVariable(v.name, "f4", coords, fill_value=0.))
                getattr(self, v.name).long_name = ""
                getattr(self, v.name).standard_name = v.name
                getattr(self, v.name).units = "unknown"
                self.user_vars += [v.name]

        self.idx = 0

    def __del__(self):
        self.dataset.close()

    def write(self, pset, time):
        """Write particle set data to file"""
        if self.type is 'array':
            self.time[:, self.idx] = time
            self.lat[:, self.idx] = np.array([p.lat for p in pset])
            self.lon[:, self.idx] = np.array([p.lon for p in pset])
            self.z[:, self.idx] = np.zeros(pset.size, dtype=np.float32)
            for var in self.user_vars:
                getattr(self, var)[:, self.idx] = np.array([getattr(p, var) for p in pset])

            self.idx += 1
        elif self.type is 'indexed':
            ind = np.arange(pset.size) + self.idx
            self.trajectory[ind] = np.array([p.trajectory for p in pset])
            self.time[ind] = time
            self.lat[ind] = np.array([p.lat for p in pset])
            self.lon[ind] = np.array([p.lon for p in pset])
            self.z[ind] = np.zeros(pset.size, dtype=np.float32)
            for var in self.user_vars:
                getattr(self, var)[ind] = np.array([getattr(p, var) for p in pset])

            self.idx += pset.size
