import numpy as np
import pylab as plt

class PSF_stack(object):
    def __init__(self):
        self.psfs = np.array((n_psfs, psf_x_size, psf_y_size), dtype=float)
        self.pos = np.array((n_psfs, 2), dtype=float)

        self.pixel_scale = 1.0
        self.wavelength = 1.0
        self.bandpass = 1.0
        self.telescope = 'Keck1'
        
        return

    def save(self):
        return


class MAOS_PSF_stack(PSF_stack):
    def __init__(self):
        super().__init__(self)

        # Ohter MAOS specific stuff.

    
    
class AIROPA_PSF_stack(PSF_stack):
    def __init__(self, directory = './',
                 psf_grid_file = 'psf_grid.fits',
                 grid_pos_file = 'grid_pos.fits'):
        
        super().__init__(self)
        
        # Load in grid of PSFs from FITS file
        hdu_psf_grid = fits.open(directory + psf_grid_file)
        self.psfs = hdu_psf_grid[0].data
        
        # Load in grid positions from FITS file
        hdu_grid_pos = fits.open(directory + grid_pos_file)
        self.pos = hdu_grid_pos[0].data
        
        # Reading in metadata (this stuff might only work for Keck NIRC2 data?)
        self.wavelength = hdu_psf_grid[0].header['EFFWAVE']
        
        max_wavelength = hdu_psf_grid[0].header['MAXWAVE']
        min_wavelength = hdu_psf_grid[0].header['MINWAVE']
        self.bandpass = max_wavelength - min_wavelength
        
        self.telescope = hdu_psf_grid[0].header['TELESCOP']
        
        # Close all HDUs
        hdu_psf_grid.close()
        hdu_grid_pos.close()
        
        return


class OOMAO_PSF_stack(PSF_stack):
    def __init__(self):
        super().__init__(self)

        # Ohter MAOS specific stuff.

    
    
