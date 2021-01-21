import numpy as np
import pylab as plt
from astropy.io import fits
import os

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
    def __init__(self,directory = './',bandpass=10,telescope='KECK1',isgrid=True, LGSpos=np.array([[-7.6,0],[0,7.6],[0.-7.6],[7.6,0]]), NGSpos=np.array([[0,5.6]]) ):
        """
        Load a grid of MAOS simulated PSFs

        Inputs
        ------
        directory : string
            A directory containing the MAOS output FITS files.
        LGSpos : numpy array
            An n by 2 array containing the x,y locations of each LGS
        NGSpos : numpy array
            An n by 2 array containing the x,y locations of each NGS
        Usage
        -----

        """
        filelist = os.listdir(directory)
        fits_files = fnmatch.filter(filelist,'evlpsfcl_1_x*_y*.fits')           #could just be *.fits?
        n_psfs = len(fits_files)
        pos = np.empty([n_psfs,2])

        first_file = True
        for i, FITSfilename in enumerate(fits_files):
            with fits.open(directory + FITSfilename) as psfFITS:
                header = psfFITS[0].header
                data = psfFITS[0].data          #shape is (y,x). Fastest changing axis (x) is printed last

            if first_file:                                              #When reading the first FITS file, initialise the arrays and read some parameters.
                psf_x_size = data.shape[1]
                psf_y_size = data.shape[0]
                psfs = np.empty([n_psfs,psf_y_size,psf_x_size])
                pixel_scale = header['dp']
                wavelength = header['wvl']*1E9                         
                first_file = False

            psfs[i,:,:] = data
            # pos[i,::-1] = [float(j) for j in FITSfilename[12:-5].split('_y')]               #read the x and y position into pos, in reversed order
            pos[i,0] = header['theta'].imag
            pos[i,1] = header['theta'].real


        super().__init__(psfs, pos, pixel_scale, wavelength, bandpass, telescope, isgrid)

        # Other MAOS specific stuff.
        self.NGSpos=NGSpos
        self.LGSpos=LGSpos
        #need a parameter to save the strehl information in some way. MAOS psfs are scaled so that the central pixel = strehl
    
        return

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

    def __init__(self, psf_strip_file, directory = './', pixel_scale=0.02081, wavelength=, bandpass=, telescope = 'Keck1',isgrid=True, LGSpos=np.array([[-7.6,0],[0,7.6],[0.-7.6],[7.6,0]]), NGSpos=np.array([[0,5.6]]) ):
        """
        Load a grid of OOMAO simulated PSFS

        Inputs
        ------
        psf_strip_file : string
            The name of the FITS file, containing a strip of PSFs stitched together.
        directory : string
            The directory containing the FITS file
        LGSpos : numpy array
            An n by 2 array containing the x,y locations of each LGS
        NGSpos : numpy array
            An n by 2 array containing the x,y locations of each NGS
        Usage
        -----


        """
        # Other OOMAO specific stuff.
        with fits.open(directory + psf_strip_file) as psfFITS:        
            header = psfFITS[0].header
            data = psfFITS[0].data
    
        psf_y_size = data.shape[0]
        psf_x_size = psf_y_size
        n_psfs = int(data.shape[1]/psf_y_size)
        grid_size = int(np.sqrt(n_psfs))
        psfs = np.empty([n_psfs,psf_y_size,psf_x_size])
        pos = np.empty([])
        for i in range(n_psfs): 
            psfs[i,:,:] = data[:,i*psf_size:(i+1)*psf_size]
            pos[i,1] = i//grid_size         #x location of psf
            pos[i,0] = i%grid_size          #y location of psf

        
        super().__init__(psfs, pos, pixel_scale, wavelength, bandpass, telescope, isgrid)
        self.NGSpos=NGSpos
        self.LGSpos=LGSpos

        return  
