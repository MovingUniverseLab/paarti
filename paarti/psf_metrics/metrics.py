"""Perform some metrics on PSF data from MAOS/AIROPA/OOMAO."""


from os import path
from astropy.io import fits
import numpy as np
from paarti import psfs



def psf_metrics(dir_label,img)

    #MAOS PSF
    maos[] = psfs.MAOS_PSF_stack()


    #AIROPA PSF
    airopa[] = psfs.AIROPA_PSF_stack()


    #OOMAO PSF
    oomao[] = psfs.OOMAO_PSF_stack()


# Data parameters
#dir_psf = '../data/'
#data_psf = '*_MAOS_psf_grid.fits'
#data_grid = '*_MAOS_grid_pos.fits'

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
