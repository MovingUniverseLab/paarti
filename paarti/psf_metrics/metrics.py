import numpy as np
import math
import astropy
import scipy
import warnings

from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy import stats
from astropy import table

from astropy.io import fits

from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry

import multiprocessing as mp


def calc_psf_metrics(psf_stack, parallel=False):
    """
    Calculate metrics on a stack of PSFs.

    Inputs:
    -------
    psf_stack : psfs.PSF_Stack object
        A PSF stack object that holds the PSFs and the positions on the
        sky of each PSF, along with plate scale, etc.

    Returns:
    --------
    mets : astropy table
        An astropy table with one row for each PSF. The computed metrics include

        'ee25' - 25% encircled-energy radius in arcsec
        'ee50' - 50% encircled-energy radius in arcsec
        'ee80' - 80% encircled-energy radius in arcsec
        'NEA'  - Noise-equivalent area in arcsec^2 (wrong?)
        'NEA2' - Noise-equivalent area in arcsec^2 (different calc, right?)
        'emp_fwhm' - Empirical FWHM in arcsec calculated based on
                     radius of circle with an area that is equivalent to the
                     area of all pixels with flux above 0.5 * max flux.
        'fwhm' - Average FWHM in arcsec from 2D Gaussian fit. 
        'xfwhm' - X FWHM in arcsec from 2D Gaussian fit. 
        'yfwhm' - Y FWHM in arcsec from 2D Gaussian fit.
        'theta' - Angle in degrees to the major axis of the 2D Gaussian.
        'ellipticity' - Ellipticity of the PSF.
        'strehl' - Strehl of the PSF, only valid for MAOS-style PSFs.
    """
    
    ####
    # Setup for parallel processing.
    ####
    if parallel:
        cpu_count = _cpu_num()
        print(f'calc_stats in parallel with {cpu_count} cores.')
    
        # Start the pool
        pool = mp.Pool(cpu_count)
        
        
    N_psfs = len(psf_stack.psfs)
    
    results_async = []

    for pp in range(N_psfs):
        psf = psf_stack.psfs[pp]

        #####
        # Add calc for this starlist to the pool.
        #####
        if parallel:
            results = pool.apply_async(calc_psf_metrics_single,
                                       (psf, psf_stack.pixel_scale))
        else:
            results = calc_psf_metrics_single(psf, psf_stack.pixel_scale)
            
        results_async.append(results)

    if parallel:
        pool.close()
        pool.join()

    ####
    # Reformat results into lists.
    ####
    ee25 = np.arange(N_psfs, dtype=float)
    ee50 = np.arange(N_psfs, dtype=float)
    ee80 = np.arange(N_psfs, dtype=float)
    NEA = np.arange(N_psfs, dtype=float)
    NEA2 = np.arange(N_psfs, dtype=float)
    emp_fwhm = np.arange(N_psfs, dtype=float)
    fwhm = np.arange(N_psfs, dtype=float)
    xfwhm = np.arange(N_psfs, dtype=float)
    yfwhm = np.arange(N_psfs, dtype=float)
    theta = np.arange(N_psfs, dtype=float)
    ellipticity = np.arange(N_psfs, dtype=float)
    strehl = np.arange(N_psfs, dtype=float)
    
    for pp in range(len(psf_stack.psfs)):
        if parallel:
            results = results_async[pp].get()
        else:
            results = results_async[pp]
        
        # Save results
        ee25[pp] = results['ee25']
        ee50[pp] = results['ee50']
        ee80[pp] = results['ee80']
        NEA[pp] = results['NEA']
        NEA2[pp] = results['NEA2']
        emp_fwhm[pp] = results['emp_fwhm']
        fwhm[pp] = results['fwhm']
        xfwhm[pp] = results['xfwhm']
        yfwhm[pp] = results['yfwhm']
        theta[pp] = results['theta']
        ellipticity[pp] = results['ellipticity']
        strehl[pp] = results['strehl']
    
    mets = table.Table([psf_stack.pos[:, 0], psf_stack.pos[:, 1],
                        ee25, ee50, ee80,
                        NEA, NEA2,
                        emp_fwhm, fwhm,
                        xfwhm, yfwhm, theta,
                        ellipticity, strehl],
                       names=('xpos', 'ypos',
                              'EE25', 'EE50', 'EE80',
                              'NEA', 'NEA2',
                              'emp_fwhm', 'fwhm',
                              'xfwhm', 'yfwhm', 'theta',
                              'ellipticity', 'strehl'),
                            meta={'name':'Metrics Table'})

    return mets

    

def calc_psf_metrics_single(psf, pixel_scale, oversamp=1, cut_radius=20):
    """
    Calculate Strehl, empirical FWHM, elliptical gaussian fit FWHM,
    encircled energy, etc. on a single PSF image. Note, the PSF image
    is assumed to be normalized in MAOS style such that the peak pixel
    value is the Strehl value. 
    
    psf : ndarray
        A 2D PSF on which to calculate metrics.
    """
    pid = mp.current_process().pid

    # Remember y is first index, x is second in the psf. e.g psf[y,x]
    
    # Assume the PSF is centered in the array.
    coords = np.array(psf.shape) / 2.0

    # Cutout and oversample the image. 
    # Odd box, with center in middle pixel.    
    psf_c = psf[int(coords[1]-cut_radius) : int(coords[1]+cut_radius+1),
                int(coords[0]-cut_radius) : int(coords[0]+cut_radius+1)]
    if oversamp > 1:
        psf_co = scipy.ndimage.zoom(psf_c, oversamp, order = 1)
        coords = np.array(psf_co.shape) / 2.0
        pixel_scale /= oversamp
    else:
        psf_co = psf

    # radial bins for the EE curves
    max_radius_pix = (psf_co.shape[0] / 2.0)  # in pixels
    max_radius_asec = max_radius_pix * pixel_scale
    
    radii_pix = np.arange(1, max_radius_pix, 1)  # in pixels
    radii_asec = radii_pix * pixel_scale

    enc_energy = np.zeros((len(radii_pix)), dtype=float)

    # Loop through radial bins and calculate EE
    for rr in range(len(radii_pix)):
        radius_pixel = radii_pix[rr]
        aperture = CircularAperture(coords, r=radius_pixel)
        phot_table = aperture_photometry(psf_co, aperture)

        energy = phot_table['aperture_sum']

        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)
        enc_energy[rr] = energy
        warnings.resetwarnings()
        warnings.filterwarnings('always', category=DeprecationWarning, append=True)

    # Normalize the encircled energy by the total. Not quite correct,
    # but close enough.
    tot_energy = psf_c.sum() * oversamp**2
    enc_energy /= tot_energy

    # Calculate the sum(PSF^2) for NEA.
    # Only do this on the last radius measurement.
    phot2_table = aperture_photometry(psf_co**2, aperture)
    int_psf2 = phot2_table['aperture_sum'][0]
    int_psf2 /= tot_energy**2   # normalize

    # Find the 50% and 80% EE values.
    # This is in oversampled pixels.
    ii25 = np.where(enc_energy >= 0.25)[0]
    if len(ii25) > 0:
        ee25_rad = radii_pix[ ii25[0] ]
    else:
        ee25_rad = np.nan
        
    ii50 = np.where(enc_energy >= 0.50)[0]
    if len(ii50) > 0:
        ee50_rad = radii_pix[ ii50[0] ]
    else:
        ee50_rad = np.nan

    ii80 = np.where(enc_energy >= 0.8)[0]
    if len(ii80) > 0:
        ee80_rad = radii_pix[ ii80[0] ]
    else:
        ee80_rad = np.nan


    # Find the median NEA in oversampled pixel^2.
    nea2 = 1.0 / int_psf2

    # Calculate the NEA in a different way. (in oversamp pixel^2)
    r_dr_2pi = 2.0 * math.pi * radii_pix[1:] * np.diff(radii_pix)
    nea = 1.0 / (np.diff(enc_energy)**2 / r_dr_2pi).sum()

    # Fit a Gaussian2D model to get FWHM and ellipticity.
    g2d_model = models.Gaussian2D(1.0, psf_co.shape[0]/2.0, psf_co.shape[1]/2.0,
                                  ee25_rad, ee25_rad, theta=0,
                                  bounds={'x_stddev':[0.1, ee80_rad],
                                          'y_stddev':[0.1, ee80_rad],
                                          'amplitude':[0.001, 2]})
    c2d_model = models.Const2D(0.0)
        
    the_model = g2d_model + c2d_model
    the_fitter = fitting.LevMarLSQFitter()

    y2d, x2d = np.mgrid[:psf_co.shape[0], :psf_co.shape[1]]

    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    g2d_params = the_fitter(the_model, x2d, y2d, psf_co)
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=UserWarning, append=True)

    # Save the FWHM and angle. In oversamp pixels.
    x_fwhm = g2d_params.x_stddev_0.value * stats.gaussian_sigma_to_fwhm
    y_fwhm = g2d_params.y_stddev_0.value * stats.gaussian_sigma_to_fwhm
    theta = np.rad2deg(g2d_params.theta_0.value % (2.0 * math.pi))

    if x_fwhm > y_fwhm:
       ellipticity = 1 - (y_fwhm / x_fwhm)
    else:
       ellipticity = 1 - (x_fwhm / y_fwhm)    
    
    # Calculate the average FWHM in oversampled pixels.
    fwhm = np.mean([x_fwhm, y_fwhm])
    
    # Find the pixels where the flux is a above half max value.
    max_flux = np.amax(psf_co) 
    half_max = max_flux / 2.0
    idx = np.where(psf_co >= half_max)
        
    # Find the equivalent circle diameter for the area of pixels.
    #    Area = pi * (FWHM / 2.0)**2 in oversamp pix^2
    area_count = len(idx[0])
    emp_FWHM = 2.0 * (area_count / np.pi)**0.5  # osamp pix

    # Calculate Strehl -- assumes peak pixel flux is Strehl (a.k.a. MAOS).
    # If not, then we really should call a proper Strehl calculator.
    strehl = psf.max()

    results = {}
    results['ee25'] = ee25_rad * pixel_scale
    results['ee50'] = ee50_rad * pixel_scale
    results['ee80'] = ee80_rad * pixel_scale
    results['NEA'] = nea * pixel_scale**2
    results['NEA2'] = nea2 * pixel_scale**2
    results['emp_fwhm'] = emp_FWHM * pixel_scale
    results['fwhm'] = fwhm * pixel_scale
    results['xfwhm'] = x_fwhm * pixel_scale
    results['yfwhm'] = y_fwhm * pixel_scale
    results['theta'] = theta   # deg
    results['ellipticity'] = ellipticity
    results['strehl'] = strehl
    
    return results


def _cpu_num():
    # How many cores to use for parallel functions
    # returns a number
    # Use N_cpu - 2 so we leave 2 for normal
    # operations. 
    cpu_count = mp.cpu_count()
    if (cpu_count > 2):
        cpu_count -= 2
        cpu_count = cpu_count if cpu_count <= 8 else 8
        
    # reurn 1 #(DEBUG)
    return cpu_count
