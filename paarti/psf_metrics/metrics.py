from os import path
import numpy as np
from paarti import psfs


#AIROPA PSF
psf_in = psfs.AIROPA_PSF_stack('c1043_psf_grid.fits', 'c1043_grid_pos.fits',
						directory='/u/skterry/work/AIROPA/sean_tests/POOR-run/fit/variable/',
						isgrid=True)



# Load data
#hdul = fits.open(path.join(dir_psf, data_psf))
#psf_data = hdul[0].data
#hdul.close()
#hdul = fits.open(path.join(dir_psf, data_grid))
#grid_data = hdul[0].data
#hdul.close()
#side_psf = psf_data.shape[1]
#n_psfs = psf_data.shape[0]
#side_psfs = int(np.sqrt(n_psfs))
