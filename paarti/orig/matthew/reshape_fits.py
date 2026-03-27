from astropy.io import fits
import numpy as np
import os

# This python script takes in the strip of simulated psfs produced by Carlos and reformats them into a 4 dimensional FITS file.
# The dimensions are [y position of psf, x position of psf, y pixel in psf, x pixel in psf.]
# By Matthew Freeman

#------PARAMETERS------
directories = ['./KAPA_PDR_PSFs/darkMatter/', 													#Location of each FITS file
				'./KAPA_PDR_PSFs/galacticCenter/',
				'./KAPA_PDR_PSFs/galaxyFormation/',
				'./KAPA_PDR_PSFs/gasGiantPlanets/']

FITSfilenames = ['PSF2D_Hband_NyqSampl_LgsAstRadius7p6arcsec_ZA30_5000iter_4LGSsquare.fits',		#Name of each data FITS file
				'PSF2D_Hband_NyqSampl_LgsAstRadius7p6arcsec_ZA50_5000iter_4LGSsquare.fits',
				'PSF2D_Hband_NyqSampl_LgsAstRadius7p6arcsec_ZA30_5000iter_4LGSsquare.fits',
				'PSF2D_Hband_NyqSampl_LgsAstRadius7p6arcsec_ZA15_5000iter_4LGSsquare.fits'
				]

strehlfilenames = ['SR_Hband_LgsAstRadius7p6arcsec_ZA30_5000iter_4LGSsquare.fits',				#Name of each strehl ratio FITS file
				'SR_Hband_LgsAstRadius7p6arcsec_ZA50_5000iter_4LGSsquare.fits',
				'SR_Hband_LgsAstRadius7p6arcsec_ZA30_5000iter_4LGSsquare.fits',
				'SR_Hband_LgsAstRadius7p6arcsec_ZA15_5000iter_4LGSsquare.fits',
				]

labels = ["dm",					#A label for the output FITS file
			"gc",
			"gf",
			"gp"
			]


TT_offsets = [[-1.1172, 0.1056],			#x,y offset of primary tip tilt star in arcseconds. For plotting later.
				[0,5.4],
				[1.7008,7.853],
				[-0.1109,0.1364]
				]	

pixel_scale = 20.81/1000 #arcseconds per pixel
psf_spacing = 2.0   #spacing between generated psfs in arcseconds
laser_radius = 7.6  #LGS asterism radius in arcseconds
output_directory = './fits_4D/'

#------UNUSED PARAMETERS------
wavelength = 1654 			#nanometers
zenith_angles = [30,50,30,15]   #degrees

def main():
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	for n in range(len(FITSfilenames)):
		reshape(directories[n],FITSfilenames[n],strehlfilenames[n],labels[n],n)


def reshape(directory, FITSfilename,strehlfilename,label,n):

	with fits.open(directory + FITSfilename) as psfFITS:
		header = psfFITS[0].header
		data = psfFITS[0].data
	with fits.open(directory + strehlfilename) as psfFITS:
		strehl_header = psfFITS[0].header
		strehl_data = psfFITS[0].data
	# with fits.open(directory + "refPsf.fits") as refFITS:
	# 	ref_header = refFITS[0].header
	# 	ref_data = refFITS[0].data

	print("Reshaping", FITSfilename)  #shape is (y,x). Fastest changing axis (x) is printed last
	# print(repr(header))

	psf_size = data.shape[0]
	number_psfs = int(data.shape[1]/data.shape[0])
	grid_size = int(np.sqrt(number_psfs))

	data4D = np.zeros([grid_size,grid_size,psf_size,psf_size])
	for i in range(grid_size):  #y value
		for j in range(grid_size):   #x value
			data4D[i,j,:,:] = data[:,(grid_size*j+i)*psf_size:(grid_size*j+i+1)*psf_size]

	print(data.shape, '->', data4D.shape )

	strehl_grid = np.zeros((grid_size,grid_size))	#reshape the strip of strehl data into a 2D grid
	for i in range (grid_size):
		for j in range(grid_size):
			strehl_grid[i,j] = strehl_data[0,grid_size*j+i]

	hdu_data = fits.PrimaryHDU(data4D)
	hdu_strehl = fits.ImageHDU(strehl_grid, name='Strehl')
	hdul = fits.HDUList([hdu_data,hdu_strehl])
	hdul[0].header['PSFSPACE'] = (psf_spacing, 'psf spacing (arcseconds)')
	hdul[0].header['PIXSCALE'] = (pixel_scale, 'pixel scale (arcseconds)')
	hdul[0].header['NGSX'] = (TT_offsets[n][0], 'NGS x offset (arcseconds)')
	hdul[0].header['NGSY'] = (TT_offsets[n][1], 'NGS x offset (arcseconds)')
	hdul[0].header['LGSRAD'] = (laser_radius, 'LGS radius (arcseconds)')
	hdul[0].header['LABEL'] = label
	hdul[0].header.comments['NAXIS1'] = 'pixel x'
	hdul[0].header.comments['NAXIS2'] = 'pixel y'
	hdul[0].header.comments['NAXIS3'] = 'psf x'
	hdul[0].header.comments['NAXIS4'] = 'psf y'		
	hdul[1].header.comments['NAXIS1'] = 'psf x'
	hdul[1].header.comments['NAXIS2'] = 'psf y'

	hdul.writeto(output_directory + label + '_simulation.fits',overwrite=True)


main()



