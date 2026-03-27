from astropy.io import fits
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


#This python script takes the 4D fits files produced by reshape_fits.py, calculates the parameters of the psf, such as 
#FWHM and EE50, and plots the results as 2D grids.
#By Matthew Freeman.

# -----PARAMETERS-----
input_directory = './fits_4D/'
output_directory = './plots/'
fit_threshold = 0.05   	#all data below this fraction of the max will be masked out for the gaussian fit.
plot_slices=False
# --------------------

def main():
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	for fits_filename in os.listdir(input_directory):
		if fits_filename[-4:] == 'fits':
			process_fits(input_directory,fits_filename)

def process_fits(directory, FITSfilename):

	with fits.open(directory + FITSfilename) as psfFITS:
		header = psfFITS[0].header
		data = psfFITS[0].data
		strehl_data = psfFITS[1].data

	print("Processing", FITSfilename)

	sim_parameters = {'label': header['LABEL'],
						'psf_spacing' : header['PSFSPACE'],
						'pixel_scale' : header['PIXSCALE'],
						'TT_offset' : [header['NGSX'],header['NGSY']],
						'laser_radius' : header['LGSRAD'],
						'psf_size' : data.shape[2],
						'grid_size' : data.shape[0],
						}



	psf_size = sim_parameters['psf_size']
	grid_size = sim_parameters['grid_size']
	psf_parameters = np.zeros((grid_size,grid_size,5)) #x_fwhm, y_fwhm, theta, 50% encircled energy, avg_fwhm
	y_coords, x_coords = np.mgrid[:psf_size,:psf_size]
	fit = fitting.LevMarLSQFitter()
	data_masked = np.ma.empty((grid_size,grid_size,psf_size,psf_size))  	#mask out data in the wings
	mask_level = np.zeros((grid_size,grid_size))

	if plot_slices and (not os.path.exists(output_directory + 'gaussian_slice/' + sim_parameters['label'])):
		os.makedirs(output_directory + 'gaussian_slice/' + sim_parameters['label'])

	for i in range(grid_size):
		for j in range(grid_size):
			mask_level[i,j] = np.max(data[i,j,:,:])*fit_threshold
			data_masked[i,j,:,:] = np.ma.masked_less(data[i,j,:,:],mask_level[i,j])
			input_params = initial_gauss_params(data[i,j,:,:])
			gauss_init = models.Gaussian2D(*input_params)
			fitted_gauss = fit(gauss_init,x_coords,y_coords,data_masked[i,j,:,:])
			psf_parameters[i,j,0] = fitted_gauss.x_fwhm
			psf_parameters[i,j,1] = fitted_gauss.y_fwhm
			psf_parameters[i,j,2] = fitted_gauss.theta.value
			psf_parameters[i,j,3] = encircle_energy(data[i,j,:,:],fitted_gauss.y_mean.value,fitted_gauss.x_mean.value)
			psf_parameters[i,j,4] = (fitted_gauss.x_fwhm + fitted_gauss.y_fwhm)/2
			if plot_slices:
				plot_slice(data,i,j,fitted_gauss,mask_level,output_directory,sim_parameters)

	fwhm_grid = psf_parameters[:,:,4] * sim_parameters['pixel_scale']
	encircled_grid = psf_parameters[:,:,3] * sim_parameters['pixel_scale']
	datasquare = np.zeros((psf_size*grid_size,psf_size*grid_size))  #reshape the 4D cube into a 2D grid
	for i in range(grid_size):
		for j in range (grid_size):
			datasquare[i*psf_size:(i+1)*psf_size,j*psf_size:(j+1)*psf_size] = data[i,j,:,:]

	plt.close('all')
	plot_psfs(datasquare, output_directory, sim_parameters)
	plot_fwhm(fwhm_grid, output_directory, sim_parameters)
	plot_ee50(encircled_grid, output_directory, sim_parameters)
	plot_strehl(strehl_data, output_directory, sim_parameters)


def initial_gauss_params(data):  #takes a psf, calculates initial parameters for the gaussian fit.
   total = data.sum()
   y1, x1 = np.indices(data.shape)  #are the order of x and y correct here?
   x2 = (x1 * data).sum() / total
   y2 = (y1 * data).sum() / total
   col = data[:, int(y2)]
   row = data[int(x2), :]
   sigmax = np.sqrt(abs(((np.arange(col.size) - y2) ** 2) * col).sum() / col.sum())
   sigmay = np.sqrt(abs(((np.arange(row.size) - x2) ** 2) * row).sum() / row.sum())
   height = np.ptp(data)
   return [height, x2, y2, sigmax, sigmay]

def encircle_energy(data,ymean,xmean):
	data = data / data.sum()
	Y, X = np.indices(data.shape)
	distance = np.sqrt((Y-ymean)**2 + (X-xmean)**2)
	trials = 201
	circles = np.zeros((trials,2))
	circles[:,0] = np.linspace(0,20,trials)   #[radius, encircled value]
	for n, r in enumerate(circles[:,0]):
		x_index, y_index = np.where(distance <= r)
		circles[n,1] = np.sum(data[x_index, y_index])
	ee50 = circles[np.argmin(np.abs(circles[:,1]-0.5)),0]
	return ee50

def plot_psfs(data_grid,output_directory,sim_parameters):
	grid_size = sim_parameters['grid_size']
	psf_spacing = sim_parameters['psf_spacing']
	laser_radius = sim_parameters['laser_radius']
	plt.figure(0)
	plt.imshow(data_grid, norm=LogNorm(vmin=np.min(data_grid), vmax=np.max(data_grid)),origin='lower', extent = (grid_size, -grid_size, -grid_size, grid_size))
	plt.plot([-laser_radius,0,0,laser_radius],[0,-laser_radius,laser_radius,0],'o',color='magenta')
	plt.plot(sim_parameters['TT_offset'][0],sim_parameters['TT_offset'][1],'or')
	plt.xlabel("East (arcseconds)")
	plt.ylabel("North (arcseconds)")
	tickpoints = np.linspace( -(grid_size-1)/2*psf_spacing , (grid_size-1)/2*psf_spacing , grid_size)
	plt.xticks(tickpoints)	
	plt.yticks(tickpoints)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels()[0::2], visible=False)
	plt.setp(ax.get_yticklabels()[0::2], visible=False)
	cbar = plt.colorbar()
	cbar.set_label("Count", rotation=90)
	plt.savefig(output_directory + sim_parameters['label'] + "_psfs.png", bbox_inches='tight', dpi=250)

def plot_fwhm(fwhm_grid,output_directory,sim_parameters):
	grid_size = sim_parameters['grid_size']
	psf_spacing = sim_parameters['psf_spacing']
	laser_radius = sim_parameters['laser_radius']
	plt.figure(1)
	plt.imshow(fwhm_grid*1000, cmap='viridis_r', origin='lower', extent = (grid_size, -grid_size, -grid_size, grid_size))
	plt.plot([-laser_radius,0,0,laser_radius],[0,-laser_radius,laser_radius,0],'o',color='magenta')
	plt.plot(sim_parameters['TT_offset'][0],sim_parameters['TT_offset'][1],'or')
	plt.xlabel("East (arcseconds)")
	plt.ylabel("North (arcseconds)")
	tickpoints = np.linspace( -(grid_size-1)/2*psf_spacing , (grid_size-1)/2*psf_spacing , grid_size)
	plt.xticks(tickpoints)	
	plt.yticks(tickpoints)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels()[0::2], visible=False)
	plt.setp(ax.get_yticklabels()[0::2], visible=False)
	cbar = plt.colorbar()
	cbar.set_label("FWHM (milliarcseconds)", rotation=90)
	plt.savefig(output_directory + sim_parameters['label'] + "_fwhm.png", bbox_inches='tight', dpi=250)

def plot_ee50(ee_grid,output_directory,sim_parameters):
	grid_size = sim_parameters['grid_size']
	psf_spacing = sim_parameters['psf_spacing']
	laser_radius = sim_parameters['laser_radius']
	plt.figure(2)
	plt.imshow(ee_grid*1000, cmap='viridis_r',origin='lower', extent = (grid_size, -grid_size, -grid_size, grid_size))
	plt.plot([-laser_radius,0,0,laser_radius],[0,-laser_radius,laser_radius,0],'o',color='magenta')
	plt.plot(sim_parameters['TT_offset'][0],sim_parameters['TT_offset'][1],'or')
	plt.xlabel("East (arcseconds)")
	plt.ylabel("North (arcseconds)")
	tickpoints = np.linspace( -(grid_size-1)/2*psf_spacing , (grid_size-1)/2*psf_spacing , grid_size)
	plt.xticks(tickpoints)	
	plt.yticks(tickpoints)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels()[0::2], visible=False)
	plt.setp(ax.get_yticklabels()[0::2], visible=False)
	cbar = plt.colorbar()
	cbar.set_label("EE50 radius (milliarcseconds)", rotation=90)
	plt.savefig(output_directory + sim_parameters['label'] + "_ee50.png", bbox_inches='tight', dpi=250)

def plot_strehl(strehl_grid,output_directory,sim_parameters):
	grid_size = sim_parameters['grid_size']
	psf_spacing = sim_parameters['psf_spacing']
	laser_radius = sim_parameters['laser_radius']
	plt.figure(3)
	plt.imshow(strehl_grid, cmap='viridis', origin='lower', extent = (grid_size, -grid_size, -grid_size, grid_size))
	plt.plot([-laser_radius,0,0,laser_radius],[0,-laser_radius,laser_radius,0],'o',color='magenta')
	plt.plot(sim_parameters['TT_offset'][0],sim_parameters['TT_offset'][1],'or')
	plt.xlabel("East (arcseconds)")
	plt.ylabel("North (arcseconds)")
	tickpoints = np.linspace( -(grid_size-1)/2*psf_spacing , (grid_size-1)/2*psf_spacing , grid_size)
	plt.xticks(tickpoints)	
	plt.yticks(tickpoints)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels()[0::2], visible=False)
	plt.setp(ax.get_yticklabels()[0::2], visible=False)
	cbar = plt.colorbar()
	cbar.set_label("Strehl ratio", rotation=90)
	plt.savefig(output_directory + sim_parameters['label'] + "_strehl.png", bbox_inches='tight', dpi=250)

def plot_slice(data,i,j,fitted_gauss,mask_level,output_directory,sim_parameters):
	psf_size = sim_parameters['psf_size']
	mid = int(psf_size/2)
	plt.figure(4)
	plt.clf()
	plt.plot([0,psf_size],[mask_level[i,j],mask_level[i,j]])
	plt.plot(data[i,j,mid,:],'.-',label="data")
	plt.plot(np.linspace(mid-10,mid+10,501),fitted_gauss(np.linspace(mid-10,mid+10,501),mid),label="fit")
	avg_fwhm = (fitted_gauss.x_fwhm + fitted_gauss.y_fwhm)/2
	ax = plt.gca()
	plt.text(0.9, 0.7,"FWHM = " + "  {:.2f}".format(avg_fwhm), horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
	plt.legend()
	plt.xlim([mid-10, mid+10])
	plt.xticks([mid-10,mid-5,mid,mid+5,mid+10])
	plt.ylim([0, np.max(data)])
	plt.savefig(output_directory + "gaussian_slice/" + sim_parameters['label'] + '/' + str(11*j+i) + ".jpg", bbox_inches='tight')	

main()



