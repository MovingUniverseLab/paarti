"""Perform some metrics on PSF data from MAOS/AIROPA/OOMAO."""


from os import path
from astropy.io import fits
import numpy as np
from paarti import psfs


# Data parameters
dir_psf = '../data/'
data_psf = 'c1049_psf_grid.fits'
data_grid = 'c1049_grid_pos.fits'

# Load data
print('Program started')
hdul = fits.open(path.join(dir_psf, data_psf))
psf_data = hdul[0].data
hdul.close()
hdul = fits.open(path.join(dir_psf, data_grid))
grid_data = hdul[0].data
hdul.close()
side_psf = psf_data.shape[1]
n_psfs = psf_data.shape[0]
side_psfs = int(np.sqrt(n_psfs))

# Process data
xs = grid_data[:, 0]
ys = grid_data[:, 1]
min_x = np.min(xs)
max_x = np.max(xs)
min_y = np.min(ys)
max_y = np.max(ys)
