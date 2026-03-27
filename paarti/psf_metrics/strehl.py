# strehl.py
# ---
# Functions to calculate Strehl ratio on a grid of PSFs
# Adapted from nirc2 strehl.py, by Jessica Lu
# Authors: Jessica Lu, Abhimat Gautam, Emily Ramey


import numpy as np

import astropy
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting

from astropy.io import fits

import os
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

import scipy, scipy.misc, scipy.ndimage
import math
import pdb

from tqdm import tqdm

from paarti import psfs

def calc_strehl(psf_stack, dl_img_file='./Keck/kp.fits', aper_size=0.3):
    
    # Extract grid of PSFs
    psf_grid = psf_stack.psfs
    (num_psfs, psf_y_size, psf_x_size) = psf_grid.shape
    
    # Empty array for storing PSF Strehls
    psf_strehls = np.empty(num_psfs)
    psf_fwhms = np.empty(num_psfs)
    psf_rms_wfes = np.empty(num_psfs)
    
    
    # Get the diffraction limited image
    scale = psf_stack.pixel_scale
    wavelength = psf_stack.wavelength
    telescope_diam = psf_stack.telescope_diam
    
    dl_img, dl_hdr = fits.getdata(dl_img_file, header=True)
    
    # Get the DL image scale and re-scale it to match the science iamge.
    if 'Keck' in dl_img_file: 
        scale_dl = 0.009952     # Hard-coded
    else:
        scale_dl = dl_hdr['PIXSCALE']
    rescale = scale_dl / scale
    
    if rescale != 1:
        dl_img = scipy.ndimage.zoom(dl_img, rescale, order=3)
    
    # Pick appropriate radii for extraction.
    # The diffraction limited resolution in pixels.
    dl_res_in_pix = 0.25 * wavelength / (telescope_diam * scale)
    # radius = int(np.ceil(2.0 * dl_res_in_pix))
    radius = int(np.ceil(aper_size / scale))
    if radius < 3:
        radius = 3
    
    # Calculate peak flux ratio, to normalize our calculated Strehl ratio
    peak_coords_dl = np.unravel_index(np.argmax(dl_img, axis=None), dl_img.shape)
    dl_peak_flux_ratio = calc_peak_flux_ratio(dl_img, peak_coords_dl, radius, skysub=False)
    
    
    # Go through each PSF in grid and calculate strehl
    print('Calculating strehls of PSFs in PSF stack')
    for cur_psf_index in tqdm(range(num_psfs)):
        cur_psf = psf_grid[cur_psf_index, :, :]
        
        
        (strehl, fwhm, rms_wfe) = calc_strehl_single(cur_psf,
                                      aper_size,
                                      wavelength,
                                      scale,
                                      telescope_diam,
                                      radius, dl_peak_flux_ratio)
        psf_strehls[cur_psf_index] = strehl
        psf_fwhms[cur_psf_index] = fwhm
        psf_rms_wfes[cur_psf_index] = rms_wfe
    
    # Save out calculated quantities to psf_stack object
    psf_stack.psf_strehls = psf_strehls
    psf_stack.psf_fwhms = psf_fwhms
    psf_stack.psf_rms_wfes = psf_rms_wfes
    
    return

def calc_strehl_single(psf, aper_size, wavelength, scale, telescope_diam,
                       radius, dl_peak_flux_ratio, coords=None):
    """
    wavelength in nm                   
    """
    
    (psf_y_size, psf_x_size) = psf.shape
    
    if coords is None:
        # Set the coordinate of the Strehl source to be center of psf grid
        coords = np.array([(psf_x_size / 2.), (psf_y_size / 2.)])
    
    coords -= 1  # Coordinates are 1 based; convert to 0 based for python
    
    # Convert wavelength to microns (used in this function) from nm (used in PAARTI)
    wavelength = wavelength * 1.e-3
    
    # The diffraction limited resolution in pixels.
    dl_res_in_pix = 0.25 * wavelength / (telescope_diam * scale)
    
    # Calculate the FWHM using a 2D gaussian fit. We will just average the two.
    # To make this fit more robust, we will change our boxsize around, slowly
    # shrinking it until we get a reasonable value.
    
    # First estimate the DL FWHM in pixels. Use this to set the boxsize for
    # the FWHM estimation... note that this is NOT the aperture size specified
    # above which is only used for estimating the Strehl.
    
    fwhm_min = 0.9 * dl_res_in_pix
    fwhm_max = 100
    fwhm = 0.0
    fwhm_boxsize = int(np.ceil((4 * dl_res_in_pix)))
    if fwhm_boxsize < 3:
        fwhm_boxsize = 3
    box_scale = 1.0
    iters = 0
    
    # Steadily increase the boxsize until we get a reasonable FWHM estimate.
    while (fwhm < fwhm_min) or (fwhm > fwhm_max):
        box_scale += iters * 0.1
        iters += 1
        g2d = fit_gaussian2d(psf, coords, fwhm_boxsize*box_scale)
        stddev = (g2d.x_stddev_0.value + g2d.y_stddev_0.value) / 2.0
        fwhm = 2.355 * stddev

        # Update the coordinates if they are reasonable. 
        if ((np.abs(g2d.x_mean_0.value - coords[0]) < fwhm_boxsize) and
            (np.abs(g2d.y_mean_0.value - coords[1]) < fwhm_boxsize)):
            
            coords = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])
    
    
    # Convert to milli-arcseconds
    fwhm *= scale * 1e3  # in milli-arseconds
    
    # Calculate the peak flux ratio
    peak_flux_ratio = calc_peak_flux_ratio(psf, coords, radius)
    
    # Normalize by the same from the DL image to get the Strehl.
    strehl = peak_flux_ratio / dl_peak_flux_ratio

    # Convert the Strehl to a RMS WFE using the Marechal approximation.
    rms_wfe = np.sqrt( -1.0 * np.log(strehl)) * wavelength * 1.0e3 / (2. * math.pi)
    
    # Check final values and fail gracefully.
    if ((strehl < 0) or (strehl > 1) or
        (fwhm > 500) or (fwhm < (fwhm_min * scale * 1e3))):
        
        strehl = -1.0
        fwhm = -1.0
        rms_wfe = -1.0

    
    return (strehl, fwhm, rms_wfe)

def calc_peak_flux_ratio(img, coords, radius, skysub=True):
    """
    img : 2D numpy array
        The image on which to calculate the flux ratio of the peak to a 
        wide-aperture.

    coords : list or numpy array, length = 2
        The x and y position of the source.

    radius : int
        The radius, in pixels, of the wide-aperture. 

    """
    # Make a cutout of the image around the specified coordinates.
    boxsize = (radius * 2) + 1
    img_cut = Cutout2D(img, coords, boxsize, mode='strict')

    # Determine the peak flux in this window.
    peak_coords_cutout = np.unravel_index(np.argmax(img_cut.data, axis=None), img_cut.data.shape)
    peak_coords = img_cut.to_original_position(peak_coords_cutout)
    peak_flux = img[peak_coords[::-1]]
    
    # Calculate the Strehl by first finding the peak-pixel flux / wide-aperture flux.
    # Then normalize by the same thing from the reference DL image. 
    aper = CircularAperture(coords, r=radius)
    aper_out = aperture_photometry(img, aper)
    aper_sum = aper_out['aperture_sum'][0]

    if skysub:
        sky_rad_inn = radius + 20
        sky_rad_out = radius + 30
        sky_aper = CircularAnnulus(coords, sky_rad_inn, sky_rad_out)
        sky_aper_out = aperture_photometry(img, sky_aper)
        sky_aper_sum = sky_aper_out['aperture_sum'][0]

        aper_sum -= sky_aper_sum

    # Calculate the peak pixel flux / wide-aperture flux
    peak_flux_ratio = peak_flux / aper_sum
    
    return peak_flux_ratio

def fit_gaussian2d(img, coords, boxsize, plot=False):
    """
    Calculate the FWHM of an objected located at the pixel
    coordinates in the image. The FWHM will be estimated 
    from a cutout with the specified boxsize.

    img : ndarray, 2D
        The image where a star is located for calculating a FWHM.
    coords : len=2 ndarray
        The [x, y] pixel position of the star in the image. 
    boxsize : int
        The size of the box (on the side), in pixels.
    """
    cutout_obj = Cutout2D(img, coords, boxsize, mode='strict')
    cutout = cutout_obj.data
    x1d = np.arange(0, cutout.shape[0])
    y1d = np.arange(0, cutout.shape[1])
    x2d, y2d = np.meshgrid(x1d, y1d)
    
    # Setup our model with some initial guess
    g2d_init = models.Gaussian2D(x_mean = boxsize/2.0,
                                 y_mean = boxsize/2.0,
                                 x_stddev=2,
                                 y_stddev=2,
                                 amplitude=cutout.max())
    g2d_init += models.Const2D(amplitude=0.0)

    fit_g = fitting.LevMarLSQFitter()
    g2d = fit_g(g2d_init, x2d, y2d, cutout)

    if plot:
        mod_img = g2d(x2d, y2d)
        plt.figure(1, figsize=(15,5))
        plt.clf()
        plt.subplots_adjust(left=0.05, wspace=0.3)
        plt.subplot(1, 3, 1)
        plt.imshow(cutout, vmin=mod_img.min(), vmax=mod_img.max())
        plt.colorbar()
        plt.title("Original")
        
        plt.subplot(1, 3, 2)
        plt.imshow(mod_img, vmin=mod_img.min(), vmax=mod_img.max())
        plt.colorbar()
        plt.title("Model")
        
        plt.subplot(1, 3, 3)
        plt.imshow(cutout - mod_img)
        plt.colorbar()
        plt.title("Orig - Mod")

        
    # Adjust Gaussian parameters to the original coordinates.
    cutout_pos = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])
    origin_pos = cutout_obj.to_original_position(cutout_pos)
    g2d.x_mean_0 = origin_pos[0]
    g2d.y_mean_0 = origin_pos[1]
    
    return g2d
    
def fwhm_to_stddev(fwhm):
    return fwhm / 2.355

def stddev_to_fwhm(stddev):
    return 2.355 * stddev

