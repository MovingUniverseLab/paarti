from astropy.io import fits
import numpy as np
import os
import fnmatch

# This python script takes a folder of simulated psfs produced by MAOS, and combines them into one 4 dimensional FITS file.
# The dimensions are [y position of psf, x position of psf, y pixel in psf, x pixel in psf.]
# By Matthew Freeman

#------PARAMETERS------
directories = ['./MAOS_output/']				#Location of each set of FITS files
labels = ["gc"]												#A label for the output FITS file	
TT_offsets = [[0,5.6]]					#x,y offset of primary tip tilt star in arcseconds. For plotting later.
# pixel_scale = 20.81/1000 	#arcseconds per pixel
laser_radius = 7.6  		#LGS asterism radius in arcseconds
output_directory = './fits_4D/'

#------UNUSED PARAMETERS------
wavelength = 1654 			#nanometers
zenith_angles = [50]   #degrees

def main():
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	for n, directory in enumerate(directories):
		combine(directory,labels[n],n)


def combine(directory,label,n):

	filelist = os.listdir(directory)
	fits_files = fnmatch.filter(filelist,'evlpsfcl_1_x*_y*.fits')
	grid_size = int(np.sqrt(len(fits_files)))
	psf_coords = np.zeros([len(fits_files),2])
	psf_index = np.zeros([len(fits_files),2])

	for m, FITSfilename in enumerate(fits_files):
		psf_coords[m] = [float(i) for i in FITSfilename[12:-5].split('_y')]

	xmax = np.max(psf_coords[:,0])
	ymax = np.max(psf_coords[:,1])
	xmin = np.min(psf_coords[:,0])
	ymin = np.min(psf_coords[:,1])

	psf_spacing = (xmax - xmin)/(grid_size-1)
	psf_index[:,0] = (psf_coords[:,0]-xmin)/psf_spacing
	psf_index[:,1] = (psf_coords[:,1]-ymin)/psf_spacing
	psf_index = psf_index.astype(int)


	data4D_created = False
	psf_size=100
	for m, FITSfilename in enumerate(fits_files):
		# print("Combining", FITSfilename)  
		
		with fits.open(directory + FITSfilename) as psfFITS:
			header = psfFITS[0].header
			data = psfFITS[0].data 			#shape is (y,x). Fastest changing axis (x) is printed last

		if not data4D_created:
			psf_size = data.shape[0]
			data4D = np.zeros([grid_size,grid_size,psf_size,psf_size])
			data4D_created = True
			pixel_scale = header['dp']

		data4D[psf_index[m,1],psf_index[m,0],:,:] = data
		# print(repr(header))

	strehl_grid = data4D[:,:,int(psf_size/2),int(psf_size/2)]

	hdu_data = fits.PrimaryHDU(data4D)
	hdu_strehl = fits.ImageHDU(strehl_grid, name='Strehl')
	hdul = fits.HDUList([hdu_data,hdu_strehl])
	hdul[0].header['PSFSPACE'] = (psf_spacing, 'psf spacing (arcseconds)')
	hdul[0].header['PIXSCALE'] = (pixel_scale, 'pixel scale (arcseconds per pixel)')
	hdul[0].header['NGSX'] = (TT_offsets[n][0], 'NGS x offset (arcseconds)')
	hdul[0].header['NGSY'] = (TT_offsets[n][1], 'NGS x offset (arcseconds)')
	hdul[0].header['LGSRAD'] = (laser_radius, 'LGS radius (arcseconds)')
	hdul[0].header['LABEL'] = (label)
	hdul[0].header.comments['NAXIS1'] = 'pixel x'
	hdul[0].header.comments['NAXIS2'] = 'pixel y'
	hdul[0].header.comments['NAXIS3'] = 'psf x'
	hdul[0].header.comments['NAXIS4'] = 'psf y'		
	hdul[1].header.comments['NAXIS1'] = 'psf x'
	hdul[1].header.comments['NAXIS2'] = 'psf y'

	hdul.writeto(output_directory + label + '_MAOS.fits',overwrite=True)


main()



