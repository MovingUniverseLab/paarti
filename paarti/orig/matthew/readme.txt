These python scripts generate plots from grids of simulated psfs.

Step 1) Convert the simulated FITS files into one 4D fits file.
			a) 	For OOMAO simulations, use reshape_fits.py.
	  	 		The OOMAO psfs are arranged in a strip. reshape_fits.py reshapes this into a single 4D FITS file.
    		b) 	For MAOS simulations, use combine_fits.py
    			MAOS outputs a separate FITS file for each PSF. combine_fits.py combines these into a single 4D FITS file
    	Modify the relevant python script to point to the folder containing the FITS files (and modify other parameters as necessary.		
    	Running the python script will create a directory (defalut ./fits_4D) containing a 4D FITS file.

Step 2) Run plot_parameters.py to read the 4D fits file and produce grid plots of the psfs, FWHM, strehl, and EE50.


The initial round of simulateds psfs are available on the google drive, under Keck > KAPA > Performance Simulations > KAPA_PDF_PSFs
The simulation parameters are described in KAON 1371. These simulations were done in OOMAO, so use reshape_fits.py.

By Matthew Freeman