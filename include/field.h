#include <stdlib.h>

typedef struct
{
  int xdim, ydim, tdim, tidx;
  float *lon, *lat;
  double *time;
  float ***data;
} CField;

/**************************************************/
/*   Interpolation routines for time and space    */
/**************************************************/

/* Local linear search to update grid index */
static inline int search_linear_float(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}

/* Local linear search to update time index */
static inline int search_linear_double(double t, int i, int size, double *tvals)
{
    while (i < size-1 && t > tvals[i+1]) ++i;
    while (i > 0 && t < tvals[i]) --i;
    return i;
}

/* Bilinear interpolation routine for 2D grid */
static inline float spatial_interpolation_bilinear(float x, float y, int i, int j, int xdim,
                                                   float *lon, float *lat, float **f_data)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[xdim] = (float (*)[xdim]) f_data;
  return (data[j][i] * (lon[i+1] - x) * (lat[j+1] - y)
        + data[j][i+1] * (x - lon[i]) * (lat[j+1] - y)
        + data[j+1][i] * (lon[i+1] - x) * (y - lat[j])
        + data[j+1][i+1] * (x - lon[i]) * (y - lat[j]))
        / ((lon[i+1] - lon[i]) * (lat[j+1] - lat[j]));
}

/* Interpolate field values in space and time.

This routine is the logical equivalent of Field.eval(time, x, y, z).
*/
static inline float field_eval(CField *f, double time, float depth, float x, float y, int xi, int yi)
{
  float (*data)[f->ydim][f->xdim] = (float (*)[f->ydim][f->xdim]) f->data;
  float f0, f1;
  double t0, t1;
  int i = xi, j = yi;
  /* Find time index for temporal interpolation */
  f->tidx = search_linear_double(time, f->tidx, f->tdim, f->time);
  /* If the time index is otside the known time range we extrapoalate */
  if (f->tidx < 0) f->tidx = 0;
  if (f->tidx >= f->tdim) f->tidx = f->tdim - 1;
  if (time != f->time[f->tidx] && f->tidx < f->tdim - 1) {
    /* Interpolate linearly in time */
    t0 = f->time[f->tidx]; t1 = f->time[f->tidx+1];
    f0 = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat, (float**)(data[f->tidx]));
    f1 = spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat, (float**)(data[f->tidx+1]));
    return f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
  } else {
    return spatial_interpolation_bilinear(x, y, i, j, f->xdim, f->lon, f->lat, (float**)(data[f->tidx]));
  }
}
