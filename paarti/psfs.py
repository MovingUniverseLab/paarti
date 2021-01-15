import numpy as np
import pylab as plt

class PSF_stack(object):
    """
    An object for a stack of PSFs (often a grid). 
    This is the general object that should be sub-classed for more
    package specific features or for loading from their files.
    """
    def __init__(self, psfs, pos, pixel_scale, wavelength, bandpass, telescope):
        """
        Create a PSF stack object for easily carrying around information needed
        to analyze and visualize a stack of PSFs.

        Inputs
        ------
        psfs : numpy array
            a 3D stack of PSFs with dimensions [N_psfs, psf_Y_size, psf_X_size]
        pos : numpy array
            a 2D array with the (y, x) sky positions for each PSF with dimensions
            [N_psfs, psf_Y_size, psf_X_size]
        pixel_scale : float
            The pixel scale of each PSFs in arcsec / pixel.
        wavelength : float
            The central wavelength of the PSFs in nm.
        bandpass : float
            The wavelength range or bandpass of the PSFs in nm.
        telescope : string
            The name of the telescope. Ideally, it would be one from the dictionary
            in paarti.telescopes.tel_names (e.g. 'Keck1').

        Usage
        -----
        """
        # 3D array with dimensions of [number of PSFs, psf_Y, psf_X]
        self.psfs = psfs

        # Give the sky positions (grid_Y, grid_X) of each PSF in the stack
        self.pos = pos

        self.pixel_scale = pixel_scale
        self.wavelength = wavelength
        self.bandpass = bandpass
        self.telescope = telescope
        
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

    
    
