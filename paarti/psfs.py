import numpy as np
import pylab as plt

from astropy.io import fits

class PSF_stack(object):
    """
    An object for a stack of PSFs (often a grid). 
    This is the general object that should be sub-classed for more
    package specific features or for loading from their files.
    """
    def __init__(self, psfs, pos, pixel_scale, wavelength, bandpass, telescope, isgrid=False):
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
    def __init__(self, psf_grid_file, grid_pos_file, directory = './',
                 pixel_scale=10, wavelength=10, bandpass=10, telescope='KeckII:NIRC2', isgrid=False):
        """
        Load up a grid of AIROPA PSFs.

        Inputs
        ------
        psf_grid_file : string
            The name of the file that contains the grid of AIROPA PSFs.
            An example is 'myimg_psf_grid.fits' output from AIROPA.
        grid_pos_file : string
            The name of the file that contains the sky positions of each
            of the AIROPA PSFs. An example is 'myimg_grid_pos.fits' from AIROPA.

        Optional Inputs
        ---------------
        directory : string
            The name of the directory to search for the files.
        isgrid : bool
            Set to true if it is a grid. Useful for plotting. 

        Usage
        ------
        mypsfs1 = psfs.AIROPA_PSF_stack('myimg_psf_grid.fits', 'myimg_grid_pos.fits', directory='./)
        mygrid = psfs.AIROPA_PSF_stack('myimg_psf_grid.fits', 'myimg_grid_pos.fits', isgrid=True)
        """
        
        # Load in grid of PSFs from FITS file
        hdu_psf_grid = fits.open(directory + psf_grid_file)
        psfs = hdu_psf_grid[0].data
        hdu_psf_grid.close()
        
        # Load in grid positions from FITS file
        hdu_grid_pos = fits.open(directory + grid_pos_file)
        pos = hdu_grid_pos[0].data
        hdu_grid_pos.close()
        
        # Call PSF_stack initializer
        super().__init__(psfs, pos, pixel_scale, wavelength, bandpass, telescope, isgrid=isgrid)

        # Any other AIROPA specific stuff to load up here?
        self.input_dir = directory
        self.file_psf_grid = psf_grid_file
        self.file_gird_pos = grid_pos_file
        
        return


class OOMAO_PSF_stack(PSF_stack):
    def __init__(self):
        super().__init__(self)

        # Ohter MAOS specific stuff.

    
    
