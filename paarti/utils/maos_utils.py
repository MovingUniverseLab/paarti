import numpy as np
import math
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
import astropy.units as u
import astropy
from paarti.psf_metrics import metrics
from paarti.utils import keck_utils
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import glob
from scipy import stats
import scipy, scipy.misc, scipy.ndimage
import matplotlib.pyplot as plt
import os
import pdb
import urllib
from pandas import read_csv
from kai import instruments
from bs4 import BeautifulSoup

strap_rmag_tab = """# File: strap_rmag.dat\n
MinMag  MaxMag  Integ   Gain	SFW 	Sky
20.0	25.0	40  	0.1 	open	1
19.0	20.0	40  	0.1 	open	1
18.0	19.0	20  	0.1 	open	1
17.0	18.0	10  	0.1 	open	1
16.0	17.0	8   	0.1 	open	1
15.0	16.0	4   	0.1 	open	1
14.0	15.0	2   	0.1 	open	1
13.0	14.0	1   	0.1 	open	1
11.0	13.0	1   	0.1 	open	0
10.0	11.0	1   	0.1 	open	0
8.5 	10.0	1   	0.1 	nd1 	0
6.0 	8.5 	1   	0.1 	nd2 	0
0.0 	6.0 	1   	0.1 	nd3 	0"""

def keck_nea_photons(m, wfs, wfs_int_time=1.0/800.0):
    """
    Calculate the number of photons, number of background photons,
    and noise equivalent angle for a natural guide star.

    Inputs:
    -------
    m : float
        Magnitude of guide star
    wfs : str
        Name of WFS to set camera properties.

    Optional Inputs:
    ----------------
    wfs_int_time : float
        Integration time of the WFS.

    Outputs:
    ------------
    SNR            : float
        Signal-to-noise ratio of a single subaperture

    sigma_theta    : float
        Noise-equivalent angle (NEA) in milliarcseconds (mas)

    Np             : float
        Number of photons (or e-) from input guide star on subaperture

    Nb             : float
        Number of photons (or e-) from background per pixel

    Notes:
    ------
    
    By Matthew Freeman and Paolo Turri, modified by Brooke DiGia

    Equations 65-68 from section 3B of Clare, R. et al (2006). 
    Adaptive optics sky coverage modelling for extremely large 
    telescopes. Applied Optics 45, 35 (8964-8978)
    """
    # List of pre-defined wave-front sensors (wfs)
    # LGSWFS-OCAM2K  : KAPA and KAPA+HODM simulation setups
    # LGS-HODM-HOWFS : KAPA+HODM+HOWFS
    wfs_list = ['LBWFS', 'LGSWFS', 'LGSWFS-OCAM2K', 'LGS-HODM-HOWFS', 
                'TRICK-H', 'TRICK-K', 'STRAP']

    if wfs not in wfs_list:
        raise RuntimeError("keck_nea_photons: Invalid WFS.")
    
    # Keck telescope diameter (m)
    D = 10.949
    # Secondary obscuration diameter (m)
    Ds = 1.8
    
    # Initialize parameters that will be set below based on input
    # wave-front sensor
    # wavelength = 0.0  # Guide star imaging wavelength
    # ps = 0.0          # Pixel scale (arcsec/px)
    # sigma_e = 0.0     # RMS detector read noise per pixel
    # theta_beta = 0.0  # Spot size on detector (rad)
    # pix_per_ap = 0    # Pixels per subaperture, for noise calculation
    
    if wfs == 'LBWFS':
        band = "R"
        wavelength = 0.641e-6

        # side length of square subaperture (m)
        side = 0.563 

        # 1.5 for WFS and low bandwithth WFS from Blake's config
        ps = 1.5
        
        # from Carlos' config
        sigma_e = 7.96
        
        # from KAON 1303 Table 16
        theta_beta = 0.49 * ( math.pi/180.0 ) / ( 60.0*60.0 )
        
        # from KAON 1303 Table 8
        throughput = 0.03
        
        # quadcell
        pix_per_ap = 4
    elif wfs == 'LGSWFS':
        band = "R"    # not actually at V
        wavelength = 0.589e-6
        
        # side length of square subaperture (m)
        side = 0.563
        
        # from Carlos' config file
        ps = 3.0
        sigma_e = 3.0
        
        # from KAON 1303 Table 20
        theta_beta = 1.5 * ( math.pi/180.0 ) / ( 60.0*60.0 )
        
        # KAON 1303 Table 8 states 0.36, but Np=1000 is already
        # measured on the detector. Modified to account for QE=0.88 
        # on the WFS detector at R-band from error budget spreadsheet
        throughput = 0.36 * 0.88
        
        # quadcell
        pix_per_ap = 4
    elif wfs == 'LGSWFS-OCAM2K':
        band = "R"
        wavelength = 0.589e-6
        
        # side length of square subaperture (m)
        side = 0.563
        
        # from Carlos' config file
        ps = 3.0
        sigma_e = 0.5
        
        # from KAON 1303 Table 20
        theta_beta = 1.5 * ( math.pi/180.0 ) / ( 60.0*60.0 )
        
        # KAON 1303 Table 8 states 0.36, but Np=1000 is already
        # measured on the detector. Modified to account for QE=0.88 
        # on the WFS detector at R-band from error budget spreadsheet
        throughput = 0.36 * 0.88
        
        # quadcell
        pix_per_ap = 4
    elif wfs == 'LGS-HODM-HOWFS':
        band = "R"
        wavelength = 0.589e-6
        side = 0.17
        ps = 3.0
        sigma_e = 0.1
        theta_beta = 1.5 * ( math.pi/180.0 ) / ( 60.0*60.0 )
        throughput = 0.36 * 0.88
        pix_per_ap = 4
    elif wfs == 'TRICK-H':
        band = "H"
        wavelength = 1.63e-6

        # side length of square subaperture (m)
        # turn into square aperture of same area as primary
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) )

        # From Carlos' config file
        ps = 0.06
        # Modified to get SNR=5 at H=15
        sigma_e = 11.188

        # Using OSIRIS FWHM from KAON 1303 Table 13 (as suggested by Peter)
        theta_beta = 0.055 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 8
        throughput = 0.56
        # Modify to add 4 lenses and a filter inside TRICK
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95
        
        # ROI reduces from 16x16 to 2x2 as residual is reduced
        pix_per_ap = 4
    elif wfs == 'TRICK-K':
        band = "K"
        wavelength = 2.19e-6

        # side length of square subaperture (m) 
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) )
        
        # From Carlos' config file
        ps = 0.04
        # Modified to get SNR=5 at H=15
        sigma_e = 11.188
        
        # Scaling the K band 0.055 by 2.19/1.63 (wavelength ratio)        
        theta_beta = 0.074 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 8
        throughput = 0.62
        # Modify to add 4 lenses and a filter inside TRICK
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95

        # ROI decreases from 16x16 to 2x2 as residual reduces
        pix_per_ap = 4
    elif wfs == 'STRAP':
        band = "R"
        wavelength = 0.641e-6

        # side length of square subaperture (m)          
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) )
        
        # From KAON 1322, just above equation 19
        ps = 1.3
        # Made up...everything seems limited by photon/background noise
        sigma_e = 0.1

        # There appears to be inconsistencies in that KAON 1322 Section 7.6
        # which quotes 3000 photons/aperture/frame (not sure what 
        # brightness star this would be for). Maybe GC R=15?
        
        # from KAON 1303 Table 16
        theta_beta = 0.49 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 7
        # Modified to account for QE=0.50 on the WFS detector at R-band
        # from error budget spreadsheet
        throughput = 0.32 #* 0.50

        # ROI
        pix_per_ap = 4

    SNR, sigma_theta, Np, Nb = keck_nea_photons_any_config(wfs,
                                                           side,
                                                           throughput,
                                                           ps,
                                                           theta_beta,
                                                           band,
                                                           sigma_e,
                                                           pix_per_ap,
                                                           wfs_int_time,
                                                           m)
    return SNR, sigma_theta, Np, Nb

def keck_nea_photons_any_config(wfs, side, throughput, ps, theta_beta,
                                band, sigma_e, pix_per_ap, time, m):
    """
    Inputs:
    ----------
    wfs         : str
        Arbitrary string name of WFS for printouts. Note there is one
        override if "LGSWFS" is in your wfs name, then it resets the
        number of background photons to 6 rather than taking the sky
        background. This is presumably from some Rayleigh backscatter
        of the laser spot.  This probably needs to be fixed.

    side        : float
        Side of a sub-aperture in meters

    throughput  : float
        Fractional throughput (0-1) of whole telescope + WFS system

    ps          : float
        Plate scale in arcsec / pixel on the WFS

    theta_beta  : float
        Spot size on sub-aperture in units of radians. For LGS spots, use
        (1.5'' * pi /180) / (60*60)

    band        : str
        Filter used for WFSing. This is used to determine the sky background
        flux contributing to each sub-aperture 

    sigma_e     : float
        Readnoise in electrons

    pix_per_ap  : int
        Number of pixels per sub-aperture

    time        : float
        Integration time of the WFS in unit of seconds

    m           : float
        Magnitude of the guide star in the specified filter

    Outputs:
    ----------
    SNR         : float
        Signal-to-noise ratio

    sigma_theta : float
        Noise-equivalent angle in milliarcsec
 
    Np          : float
        Number of photons from the star per pixel within
        subaperture

    Nb          : float
        Number of background photons per pixel within
        subaperture
    """
    print('Assumptions:')
    print(f'  Wave-Front Sensor       = {wfs}')
    print(f'  Pupil Aperture Diameter = {side:.2f} m (assumed square)')
    print(f'  Throughput (w QE)       = {throughput:.2f}')
    print(f'  Plate Scale             = {ps:.3f} arcsec/pix')
    print(f'  Spot Size Diameter      = {theta_beta*206265:.3f} arcsec')
    print(f'  Filter                  = {band}')
    print(f'  Readnoise               = {sigma_e} e-')
    print(f'  Pixels per Subaperture  = {pix_per_ap}')
    print(f'  Integration Time        = {time:.4f} s')
    print(f'  Guide Star Magnitude    = {m:.2f}')
    print()
    
    # Calculate number of photons and background photons
    Np, Nb = n_photons(side, time, m, band, ps, throughput)

    """
    # Fix LGS background
    if 'LGSWFS' in wfs:
        # Convert 6 background photons per subaperture to
        # background per pixel (4 pixels per subaperture,
        # quadcell)
        Nb = 6.0 * (1.0/4.0)
    """

    # # area of supaperture        
    # A_sa = side**2
    # # total number of subapertures for the NGS WFS
    # N_sa = A_sa/(math.pi*(D/2)**2)
    # # Effective spot size of the subaperture NGS assuming
    # # seeing limited image. (eq 67).
    # theta_beta = wavelength/(4*r_0*0.4258)
    # # Effective spot size of the subaperture NGS assuming a
    # # diffraction limited core. (eq 68)    
    # theta_beta = 3*math.pi*wavelength*np.sqrt(N_sa)/(16*D)
    # signal to noise ratio of a single subaperture (eq 66)
    
    SNR = Np / np.sqrt(Np + pix_per_ap*Nb + pix_per_ap*sigma_e**2)

    # Noise equivalent angle in milliarcseconds (eq 65)
    sigma_theta = theta_beta/SNR  * ( 180.0/math.pi ) * 60.0 * 60.0 * 1000.0

    print('Outputs:')
    print(f"  N_photons from star (powfs.siglev for MAOS config): {Np:.3f}")
    print(f"  N_photons per pixel from background (powfs.bkgrnd):   {Nb:.3f}")
    print(f"  SNR:                                   {SNR:.3f}")
    print(f"  NEA (powfs.nearecon): {sigma_theta:.3f} mas")
    
    return SNR, sigma_theta, Np, Nb
    
def n_photons(side, time, m, band, ps, throughput):
    """
    Calculate the number of photons from a star and
    background incident on a square area in a given time 
    interval.

    By Paolo Turri
        
    Bibliography:
    [1] Bessel et al. (1998)
    [2] Mann & von Braun (2015)
    [3] https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_%28Sec_2%29.html

    Inputs:
    ------------
    side       : float
        Side of square aperture (m)

    time       : float
        Time interval (s)

    m          : float
        Apparent magnitude (Vega system)

    band       : string
        Band name ("U", "B", "V", "R", "I", "J", "H", "K")
        
    ps         : float
        Pixel scale (arcsec/px)

    throughput : float
        Throughput with quantum efficiency

    Outputs:
    ------------
    n_ph_star  : float
        Number of star photons

    n_ph_bkg   : float
        Number of background photons (px^-1)
    """
    # Fixed parameters
    c = 2.99792458e8   # Speed of light (m s^-1)
    h = 6.6260755e-27  # Plank constant (erg s)
    # Bands' names, effective wavelengths (microns), equivalent widths
    # (microns), fluxes (10^-11 erg s^-1 cm^-2 A^-1) and background
    # in (magnitudes arcsec^-2) [1, 2, 3].
    bands = {'name': ["U", "B", "V", "R", "I", "J", "H", "K"],
             'lambd': [0.366, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19],
             'delta_lambd': [0.0665, 0.1037, 0.0909, 0.1479, 0.1042, 0.3268, 0.2607,
                             0.5569],
             'phi_erg': [417.5, 632, 363.1, 217.7, 112.6, 31.47, 11.38, 3.961],
             'bkg_m': [21.6, 22.3, 21.1, 20.3, 19.2, 14.8, 13.4, 12.6]}

    # Get band's data
    band_idx = np.where(np.array(bands['name']) == band)[0][0]
    # Band effective wavelength (microns)    
    lambd = float(bands['lambd'][band_idx])
    # Band equivalent width (microns)
    delta_lamb = float(bands['delta_lambd'][band_idx])
    # Flux (erg s^-1 cm^-2 A^-1)
    phi_erg = float(bands['phi_erg'][band_idx])
    # Background magnitude (arcsec^-2)    
    bkg_m = float(bands['bkg_m'][band_idx])

    # Band frequency (Hz)
    f = c / (lambd * 1e-6)
    # Numeric flux (s^-1 cm^-2 A^-1)
    phi_n = phi_erg * 1e-11 / ( h * f )
    # Zeropoint (m = 0) number of photons on detector
    n_ph_0 = phi_n * ( (side * 1e2) ** 2) * time * delta_lamb * 1e4 * throughput  
    # Number of star photons
    n_ph_star = n_ph_0 * ( 10**(-0.4 * m) ) 

    # Number of background photons (px^-1)
    n_ph_bkg = n_ph_0 * ( 10**(-0.4 * bkg_m) ) * (ps**2)  
    
    return n_ph_star, n_ph_bkg

def keck_ttmag_to_itime(ttmag, wfs='strap'):
    """
    Calculate the expected integration time for STRAP given
    a tip-tilt star magnitude in the R-band.

    Inputs:
    ------------
    ttmag : float
        Tip-tilt star brightness in apparent R-band magnitudes in
        the Vega system

    Outputs:
    ------------
    itime : float
        The integration time used for STRAP in seconds
    """
    if wfs == 'strap':
        tab = Table.read(strap_rmag_tab, format='ascii')
    else:
        raise RuntimeError(f'Invalid WFS type: {wfs}')

    # Find the bin where our TT star belongs.
    idx = np.where((tab['MinMag'] <= ttmag) & (ttmag < tab['MaxMag']))[0]

    # Fetch the integration time. 
    itime = tab['Integ'][idx[0]]
    
    return itime

def print_wfe_metrics(directory='./', seed=10):
    """
    Function to print various wave-front error (WFE) metrics 
    to terminal.

    Inputs:
    ------------
    directory      : string, default is current directory
        Path to directory where simulation results live

    seed           : int, default=10
        Seed with which simulation was run

    Outputs:
    ------------
    open_mean_nm   : array, len=3, dtype=float
        Array containing WFE metrics for open-loop MAOS results

    closed_mean_nm : array, len=3, dtype=float
        Array containing WFE metrics for closed-loop MAOS results
    """
    # Import the MAOS readbin function
    import readbin
    
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    print("Looking in directory:", directory)

    # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    open_mean_nm = np.sqrt(results[0].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    clos_mean_nm = np.sqrt(results[2].mean(axis=0)) * 1.0e9

    print('---------------------')
    print('WaveFront Error (nm): [note, piston removed from all]')
    print('---------------------')
    print(f'{"      ":<7s}  {"Total":>11s}  {"High_Order":>11s}  {"TT":>11s}')
    print(f'{"Open  ":<7s}  {open_mean_nm[0]:11.1f}  {open_mean_nm[2]:11.1f}  {open_mean_nm[1]:11.1f}')
    print(f'{"Closed":<7s}  {clos_mean_nm[0]:11.1f}  {clos_mean_nm[2]:11.1f}  {clos_mean_nm[1]:11.1f}')

    return open_mean_nm, clos_mean_nm
    
def print_psf_metrics_x0y0(directory='./', oversamp=3, seed=10):
    """
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths. Closed-loop.

    Inputs:
    ------------
    directory        : string, default is current directory
        Directory where MAOS simulation results live

    oversamp         : int, default=3

    seed             : int, default=10
        Simulation seed (seed value for which MAOS simulation was run)

    Outputs:
    ------------
    wavelengths      : array, dtype=float
        Array of wavelengths for which MAOS simulation was run and for
        which output metrics were calculated

    strehl_values    : array, dtype=float
        Array of Strehl values for each wavelength

    fwhm_gaus_values : array, dtype=float
        Array of FWHM values for Gaussians fit to each MAOS PSF at
        each wavelength

    fwhm_emp_values  : array, dtype=float
        Array of empirical FWHM values for each MAOS PSF. Empirical
        FWHM is calculated by locating the pixel with the largest flux,
        dividing that flux by 2, finding the nearest pixel with this halved 
        flux value, and computing the distance between them. This quantity
        is then converted to micro-arcsec (mas) using the MAOS pixel scale
        (arcsec/px) from the MAOS PSF header

    r_ee80_values    : array, dtype=float
        Array of radii for each MAOS PSF. At each wavelength, a radius
        is computed on the MAOS PSF, within which 80% of the total
        image flux is contained.
    """
    print("Looking in %s for simulation results..." % directory)  
    fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x0_y0.fits')
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)

    wavelengths = np.zeros(nwvl)
    strehl_values = np.zeros(nwvl)
    fwhm_gaus_values = np.zeros(nwvl)
    fwhm_emp_values = np.zeros(nwvl)
    r_ee80_values = np.zeros(nwvl)
 
    print(f'{"Wavelength":10s} {"Strehl":>6s} {"FWHM_gaus":>10s} {"FWHM_emp":>10s} {"r_EE80":>6s}')
    print(f'{"(microns)":10s} {"":>6s} {"(mas)":>10s} {"(mas)":>10s} {"(mas)":>6s}')
    
    for pp in range(nwvl):
        psf = psf_all_wvls[pp].data
        hdr = psf_all_wvls[pp].header
        mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], 
                                               oversamp=oversamp)
        wavelengths[pp] = hdr["WVL"] * 1.0e6
        strehl_values[pp] = mets["strehl"]
        fwhm_gaus_values[pp] = mets["emp_fwhm"] * 1.0e3
        fwhm_emp_values[pp] = mets["fwhm"] * 1.0e3
        r_ee80_values[pp] = mets["ee80"] * 1.0e3

        sout  = f'{hdr["WVL"]*1e6:10.3f} '
        sout += f'{mets["strehl"]:6.2f} '
        sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["ee80"]*1e3:6.1f}' 
        print(sout)

    psf_all_wvls.close()
    return wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee80_values

def print_psf_metrics_open(directory='./', oversamp=3, seed=10):
    """
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths. Open-loop.

    Inputs:
    ------------
    directory : string, default is current directory
        Directory where MAOS simulation results live

    oversamp  : int, default=3

    seed      : int, default=10
        Seed corresponding to MAOS simulation (same value that
        was given to MAOS to run the simulation)
    Outputs:
    ------------
    None, prints to terminal
    """

    fits_files = glob.glob(directory + f'evlpsfol_{seed}.fits')
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)
 
    print(f'{"Wavelength":10s} {"Strehl":>6s} {"FWHM_gaus":>10s} {"FWHM_emp":>10s} {"r_EE80":>6s}')
    print(f'{"(microns)":10s} {"":>6s} {"(mas)":>10s} {"(mas)":>10s} {"(mas)":>6s}')
    
    for pp in range(nwvl):
        psf = psf_all_wvls[pp].data
        hdr = psf_all_wvls[pp].header
        mets = metrics.calc_psf_metrics_single(psf, hdr['DP'],
                                               oversamp=oversamp)
        sout  = f'{hdr["WVL"]*1e6:10.3f} '
        sout += f'{mets["strehl"]:6.2f} '
        sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["ee80"]*1e3:6.1f}' 
        print(sout)

    psf_all_wvls.close()
    return

def read_maos_psd(psd_input_file, type='jitter'):
    """
    Read in a MAOS PSD file and return the frequency and PSD arrays
    with units. Note, there are two types...a "jitter" file usually
    has input for windshake and vibrations, which is in units of
    radian^2/Hz. The second is a the residual WFE PSD output by
    MAOS in units of m^2/Hz.

    Inputs:
    ------------
    psd_input_file : string
        Path to input PSD file

    type           : string, default='jitter'
        Type of input PSD file

    Outputs:
    ------------
    freq           : array, dtype=float
        Frequency array with units attached

    psd            : array, dtype=float
        PSD array with units attached
    """
    import readbin
    
    if psd_input_file.endswith('fits'):
        psd_in = fits.getdata(psd_input_file)
    else:
        psd_in = readbin.readbin(psd_input_file)

    freq = psd_in[0] * u.Hz

    if type == 'jitter':
        psd = psd_in[1] * u.radian**2 / u.Hz
    else:
        psd = psd_in[1] * u.m**2 / u.Hz

    return freq, psd
    
def psd_add_vibrations(psd_input_file, vib_freq, vib_jitter_amp):
    """
    Take an input temporal power-spectral density (PSD) function in rad^2/Hz
    (as expected for MAOS) and modify it to add a vibration peak with
    a log-normal distribution peaked at the input vibration frequency
    and with an integrated jitter amplitude equal to the specified value
    in arcseconds.

    Inputs:
    ------------
    psd_input_file : string
        The name of the input PSD file for windshake and vibrations in 
        MAOS format. This is a 2 column binary array with the first column
        containing frequency in Hz and the second column containing the
        PSD in radian^2 / Hz. The file should be readable by the MAOS
        readbin utility or it should be a FITS file

    vib_freq       : float
        Peak vibration frequency in Hz

    vib_jitter_amp : float
        Integrated jitter in arcsec over the whole vibration peak

    Outputs:
    ------------
    freq           : array, dtype=float
        Frequency (Hz) array

    psd            : array, dtype=float
        Modified power-spectral density array (rad^2/Hz)
    """
    freq, psd = read_maos_psd(psd_input_file)
    dfreq = np.diff(freq)

    # Create a vibration peak
    vib_model = stats.lognorm(0.1, scale=vib_freq, loc=1)
    vib_model_psd = vib_model.pdf(freq) * u.radian**2 / u.Hz

    # Normalize the vibration peak to have a total jitter as specified
    norm = np.sqrt(np.sum(vib_model_psd[1:] * dfreq)).to('arcsec')
    vib_model_psd *= (vib_jitter_amp * u.arcsec / norm)**2

    psd += vib_model_psd
    return freq, psd
    
def psd_integrate_sqrt(freq, psd):
    """
    Function to integrate and take the square root of a PSD to give the total 
    WFE or jitter.

    Inputs:
    ------------
    freq      : float
        Frequency (Hz) array

    psd       : float
        Power-spectral density array (rad^2/Hz)

    Outputs:
    ------------
    total_rms : float
        Total RMS WFE
    """
    dfreq = np.diff(freq)
    total_variance = scipy.integrate.trapezoid(psd, freq)

    # math.sqrt changed to np.sqrt since that seems to have better
    # support for Astropy Quantities (units attached) -- Brooke D.
    total_rms = np.sqrt(total_variance)
    return total_rms

def calc_strehl(sim_dir, out_file, skysub=False, sim_seed=1, apersize=0.3):
    """
    Modified from KAI by Brooke DiGia
    (https://github.com/Keck-DataReductionPipelines/KAI/tree/dev)
    for use on MAOS-generated PSF files.

    Function to calculate the Strehl, root-mean-square (RMS) wave-front error
    (WFE), and full-width at half-maximum (FWHM) of a MAOS PSF stack.

    Inputs:
    ------------
    sim_dir          : str
        The directory where the MAOS simulation results live

    sim_seed         : int, default = 1
        The seed passed to MAOS for running the simulation

    out_file         : str
        The name of the output text file

    skysub           : boolean, default = False
        True to perform sky subtraction on PSF. Should be False for MAOS
        PSFs

    aper_size        : float, default = 0.3 arcsec
        The aperture size over which to calculate the Strehl and FWHM

    Outputs:
    ------------
    strehl_to_return : array, len(nwvl), dtype=float
        Strehl values at each wavelength for which MAOS was run

    fwhm_to_return   : array, len(nwvl), dtype=float
        FWHM values at each wavelength

    rmswfe_to_return : array, len(nwvl), dtype=float
        RMS WFE values at each wavelength

    Function writes to user-specified output text file and prints
    results to terminal
    """
    # Setup the output file and format.
    _out = open(out_file, 'w')

    fmt_hdr = '{img:<3s} {strehl:>10s} {rms:>7s} {fwhm:>7s}\n'
    fmt_dat = '{img:<3f} {strehl:10.6f} {rms:7.1f} {fwhm:7.2f}\n'
    
    _out.write(fmt_hdr.format(img='Wavelength', strehl='Strehl', rms='RMSwfe', 
                              fwhm='FWHM'))
    _out.write(fmt_hdr.format(img='(micron)', strehl='()', rms='(nm)', 
                              fwhm='(mas)'))

    # Get the PSF .fits files from the input simulation directory
    path = sim_dir + f'evlpsfcl_{sim_seed}_x0_y0.fits'
    print(path)
    fits_files = glob.glob(path)
    print(fits_files)
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)

    # Get the diffraction limited image from the input simulation directory
    dl_img_files = glob.glob(sim_dir + 'evlpsfdl.fits')
    dl_all_wvls = fits.open(dl_img_files[0])

    # Loop over the MAOS PSF stack to calculate metrics
    print("Lambda (micron) | Strehl | RMS WFE (nm) | FWHM (mas)")
    print("----------------------------------------------------------------")
    strehl_to_return = np.zeros(nwvl)
    fwhm_to_return = np.zeros(nwvl)
    rmswfe_to_return = np.zeros(nwvl)
    for i in range(nwvl):
        # Pull current PSF and DL image from stack
        psf = psf_all_wvls[i].data
        hdr = psf_all_wvls[i].header
        dl_img = dl_all_wvls[i].data
        dl_hdr = dl_all_wvls[i].header

        # DL wavelength (microns) and pixel scale (arcsec/px)
        dl_lambda = dl_hdr["WVL"]*1e6
        scale = dl_hdr["DP"]

        # PSF wavelength (microns)
        psf_lambda = hdr["WVL"]*1e6

        # Check that DL wavelength and PSF wavelength are matched
        if dl_lambda != psf_lambda:
            print("Error: PSF wavelength does not match DL wavelength.")
            _out.close()
            return

        # Pick appropriate extraction radius
        radius = int(np.ceil(apersize / scale))
        # In case the computed radius is too small...
        if radius < 3:
            radius = 3

        # Perform some wide-aperture photometry on the 
        # diffraction-limited image
        peak_coords_dl = np.unravel_index(np.argmax(dl_img, axis=None), 
                                          dl_img.shape)

        # Calculate the peak flux ratio
        try:
            dl_peak_flux_ratio = calc_peak_flux_ratio(sim_dir, dl_img, 
                                                      peak_coords_dl, 
                                                      radius, dl_lambda, 
                                                      skysub)
        except astropy.nddata.PartialOverlapError:
            print("astropy.nddata.PartialOverlapError")
            _out.close()
            return

        # Calculate Strehl, FWHM, RMS WFE
        strehl, fwhm, rmswfe = calc_strehl_single(sim_dir, psf, hdr, 
                                                  radius, skysub, 
                                                  dl_peak_flux_ratio)
        strehl_to_return[i] = strehl
        fwhm_to_return[i] = fwhm
        rmswfe_to_return[i] = rmswfe

        _out.write(fmt_dat.format(img=psf_lambda, strehl=strehl, 
                                  rms=rmswfe, fwhm=fwhm))
        print(fmt_dat.format(img=psf_lambda, strehl=strehl, 
                             rms=rmswfe, fwhm=fwhm), end="")
        print(">> PSF peak flux value: %0.6f\n" % psf.max())

        """
        # For plotting purposes:
        if psf_lambda == 2.12:
            plt.figure()
            plt.imshow(dl_img)
            plt.savefig("/u/bdigia/work/ao/single_psfs/good_run_psfs/test_maos_dl_plot.pdf")
        """

    # Close output file
    _out.close()
    return strehl_to_return, fwhm_to_return, rmswfe_to_return

def calc_strehl_single(sim_dir, psf, hdr, radius, skysub, dl_peak_flux_ratio):
    """
    Modified from KAI by Brooke DiGia 
    (https://github.com/Keck-DataReductionPipelines/KAI/tree/dev)
    for use on MAOS-generated PSFs and diffraction-limited images.

    Function to calculate the Strehl for a single image PSF.

    Inputs:
    ----------
    sim_dir : string 
        Simulation directory, output will be stored here

    psf     : array
        PSF data

    hdr     : dictionary 
        FITS header associated with PSF data

    radius  : float
        Extraction radius in fitting

    skysub  : boolean 
        True to perform sky subraction on PSF

    Outputs:
    ------------
    strehl  : float
        Calculated Strehl value

    fwhm    : float
        Full-width at half-maximum of Gaussian fitted to input PSF

    rms_wfe : float
        Root-mean-square wave-front error (nm)
    """
    wavelength = hdr["WVL"]*1e6 # microns
    scale = hdr["DP"]           # arcsec/px

    # Coordinates of Strehl source (MAOS PSFs are output such that the
    # Strehl source is always centered in the image)
    coords = np.array([psf.shape[0]/2.0 , psf.shape[1]/2.0])

    # First estimate the DL FWHM in pixels. Use this to set the initial boxsize 
    # for the FWHM estimation...note that this is NOT the aperture size 
    # specified above, which is only used for estimating the Strehl:
    
    # Keck telescope diameter in meters
    telescope_diam = 10.5
    dl_res_in_pix = ( 0.25 * wavelength ) / ( telescope_diam * scale )
    fwhm_min = 0.9 * dl_res_in_pix
    fwhm_max = 100.0
    fwhm = 0.0
    fwhm_boxsize = int( np.ceil( ( 4 * dl_res_in_pix ) ) )
    if fwhm_boxsize < 3:
        fwhm_boxsize = 3
    pos_delta_max = 2 * fwhm_min
    box_scale = 1.0
    iters = 0
    
    # Steadily increase the boxsize until we get a reasonable FWHM
    while ( (fwhm < fwhm_min) or (fwhm > fwhm_max) ) and (iters < 30):
        box_scale += iters * 0.1
        iters += 1
        g2d = fit_gaussian2d(psf, coords, fwhm_boxsize * box_scale, 
                             fwhm_min=0.8 * fwhm_min, fwhm_max=fwhm_max,
                             pos_delta_max=pos_delta_max)
        sigma = (g2d.x_stddev_0.value + g2d.y_stddev_0.value) / 2.0
        fwhm = stddev_to_fwhm(sigma)

        # Update the coordinates if they are reasonable. 
        if ((np.abs(g2d.x_mean_0.value - coords[0]) < fwhm_boxsize) and
            (np.abs(g2d.y_mean_0.value - coords[1]) < fwhm_boxsize)):
            coords = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])

    # Convert to milli-arcseconds
    fwhm *= scale * 1e3

    # Calculate the peak flux ratio
    peak_flux_ratio = calc_peak_flux_ratio(sim_dir, psf, coords, radius, 
                                           wavelength, skysub)
    # Normalize by the same from the DL image to get the Strehl
    strehl = peak_flux_ratio / dl_peak_flux_ratio

    # Convert the Strehl to a RMS WFE using the Marechal approximation
    rms_wfe = np.sqrt( -1.0 * np.log(strehl) ) * (wavelength * 1.0e3) / ( 2.0 * math.pi )
    
    # Check final values and fail gracefully.
    if ((strehl < 0) or (strehl > 1) or
        (fwhm > 500) or (fwhm < (fwhm_min * scale * 1e3))):
        
        strehl = -1.0
        fwhm = -1.0
        rms_wfe = -1.0

    return strehl, fwhm, rms_wfe

def calc_peak_flux_ratio(sim_dir, img, coords, radius, wavelength, skysub):
    """
    Modified from KAI by Brooke DiGia for use on MAOS-generated PSFs.
    Function to calculate the ratio of peak flux in the input PSF image
    to the sum of the PSF pixel values. Optional plotting routine to
    visually inspect the sky subtraction annulus (if skysub is turned on).

    Inputs:
    ------------
    sim_dir         : string
        Simulation directory

    img             : 2D numpy array
        The image on which to calculate the flux ratio of the peak to a 
        wide-aperture

    coords          : list or numpy array, length = 2
        The x and y position of the source

    radius          : int
        The radius, in pixels, of the wide-aperture 

    wavelength      : float
        Wavelength of the associated img/PSF for plotting purposes

    skysub          : boolean
        True to perform sky subtraction on PSF

    Outputs:
    ------------
    peak_flux_ratio : float
        Peak flux ratio
    """
    # Determine the peak flux
    peak_coords = np.unravel_index(np.argmax(img.data, axis=None), 
                                             img.data.shape)
    peak_flux = img[peak_coords]
    
    # Calculate the Strehl by first finding the peak-pixel flux / 
    # wide-aperture flux. Then normalize by the same thing from 
    # the reference DL image. 
    aper_sum = np.sum(img)

    if skysub:
        sky_rad_inn = radius + 20
        sky_rad_out = radius + 30
        sky_aper = CircularAnnulus(coords, sky_rad_inn, sky_rad_out)
        sky_aper_out = aperture_photometry(img, sky_aper)
        sky_aper_sum = sky_aper_out['aperture_sum'][0]

        aper_sum -= sky_aper_sum
        plt.imshow(img, aspect="auto")
        annulus_patches = sky_aper.plot(color="red", label="Annulus")
        plt.title("MAOS PSF at %0.2f microns" % wavelength)
        plt.savefig("%s/PSF_annuli_%0.2f_microns.pdf" % 
                    (sim_dir, wavelength))

    # Calculate the peak pixel flux / wide-aperture flux
    peak_flux_ratio = peak_flux / aper_sum
    return peak_flux_ratio

def fit_gaussian2d(img, coords, boxsize, plot=False, fwhm_min=1.7, 
                   fwhm_max=30, pos_delta_max=1.7):
    """
    Calculate the FWHM of an objected located at the pixel
    coordinates in the image. The FWHM will be estimated 
    from a cutout with the specified boxsize. Adopted from
    the KAI repository (linked in functions above).

    Inputs:
    ------------
    img           : ndarray, 2D
        The image where a star is located for calculating a FWHM

    coords        : len=2 ndarray
        The [x, y] pixel position of the star in the image

    boxsize       : int
        The size of the box (on the side), in pixels

    fwhm_min      : float, optional
        The minimum allowed FWHM for constraining the fit (pixels)

    fwhm_max      : float, optional
        The maximum allowed FWHM for constraining the fit (pixels)

    pos_delta_max : float, optional
        The maximum allowed positional offset for constraining the fit (px)
        This ensures that the fitter doesn't wander off to a bad pixel

    Outputs:
    ------------
    g2d           : Gaussian model object
        2D Gaussian fit
    """
    cutout_obj = Cutout2D(img, coords, boxsize, mode='strict')
    cutout = cutout_obj.data
    x1d = np.arange(0, cutout.shape[0])
    y1d = np.arange(0, cutout.shape[1])
    x2d, y2d = np.meshgrid(x1d, y1d)
    
    # Setup our model with some initial guess
    x_init = boxsize/2.0
    y_init = boxsize/2.0
    
    x_init = np.unravel_index(np.argmax(cutout), cutout.shape)[1]
    y_init = np.unravel_index(np.argmax(cutout), cutout.shape)[0]
    
    stddev_init = fwhm_to_stddev(fwhm_min)
    
    g2d_init = models.Gaussian2D(x_mean = x_init,
                                 y_mean = y_init,
                                 x_stddev = stddev_init,
                                 y_stddev = stddev_init,
                                 amplitude=cutout.max())
    g2d_init += models.Const2D(amplitude=0.0)
    g2d_init.x_stddev_0.min = fwhm_to_stddev(fwhm_min)
    g2d_init.y_stddev_0.min = fwhm_to_stddev(fwhm_min)
    g2d_init.x_stddev_0.max = fwhm_to_stddev(fwhm_max)
    g2d_init.y_stddev_0.max = fwhm_to_stddev(fwhm_max)
    
    g2d_init.x_mean_0.min = x_init - pos_delta_max
    g2d_init.x_mean_0.max = x_init + pos_delta_max
    g2d_init.y_mean_0.min = y_init - pos_delta_max
    g2d_init.y_mean_0.max = y_init + pos_delta_max
    
    fit_g = fitting.LevMarLSQFitter()
    g2d = fit_g(g2d_init, x2d, y2d, cutout)
    
    if plot:
        mod_img = g2d(x2d, y2d)
        plt.figure(1, figsize=(15,5))
        plt.clf()
        plt.subplots_adjust(left=0.05, wspace=0.3)
        plt.subplot(1, 3, 1)
        plt.imshow(cutout, vmin=mod_img.min(), vmax=mod_img.max(),
                   origin='lower')
        plt.colorbar()
        plt.title("Original")
        
        plt.subplot(1, 3, 2)
        plt.imshow(mod_img, vmin=mod_img.min(), vmax=mod_img.max(),
                   origin='lower')
        plt.colorbar()
        plt.title("Model")
        
        plt.subplot(1, 3, 3)
        plt.imshow(cutout - mod_img, origin='lower')
        plt.colorbar()
        plt.title("Orig - Mod")
        
        plt.show()
        
        # plt.show(block=0)
        # plt.savefig('strehl_fit.pdf')
        # pdb.set_trace()
        
    # Adjust Gaussian parameters to the original coordinates.
    cutout_pos = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])
    origin_pos = cutout_obj.to_original_position(cutout_pos)
    g2d.x_mean_0 = origin_pos[0]
    g2d.y_mean_0 = origin_pos[1]
    
    return g2d
    
def fwhm_to_stddev(fwhm):
    """
    Function to convert input full-width at half-maximum to standard
    deviation, assuming a Gaussian distribution.

    Inputs:
    ------------
    fwhm  : float
        Full-width at half-maximum

    Outputs:
    ------------
    sigma : float
        Standard deviation
    """
    sigma = fwhm / ( 2.0 * math.sqrt( 2.0 * math.log(2.0) ) )
    return sigma

def stddev_to_fwhm(stddev):
    """
    Function to convert input standard deviation to full-width at
    half-maximum, assuming a Gaussian distribution.

    Inputs:
    ------------
    stddev : float
        Standard deviation

    Outputs:
    ------------
    fwhm   : float
        Full-width at half-maximum
    """
    fwhm = 2.0 * math.sqrt( 2.0 * math.log(2.0) ) * stddev
    return fwhm 

def fried(DIMM, wvl=500):
    """
    Function to calculate the Fried parameter r0z given the total seeing
    in arcseconds.

    Inputs:
    ------------
    DIMM : float
        DIMM seeing, arcsec 
    
    wvl  : float
        Wavelength in nm, default is 500 nm

    Outputs:
    ------------
    r0z  : float
        The Fried parameter, r0z, in meters

    By Brooke DiGia
    """ 
    r0z = 0.98 * ( wvl*1e-9 / arcsec_to_rad(DIMM) )
    return r0z

def arcsec_to_rad(x):
    """
    Function to convert input quantity from arcseconds to radians.

    Inputs:
    ------------
    x : float 
        Quantity to be converted from arcsec to radians

    Outputs:
    ------------
    x : float
        Desired quantity in radians

    By Brooke DiGia
    """
    return x * (1.0/3600.0) * (math.pi/180.0)

def estimate_turbulence(dimm, mass_wts, date, plot, wvl=500, 
                        normalize=True):
    """
    Based on equations 16-19 in KAON496:
    https://www.oir.caltech.edu/twiki_oir/pub/Keck/NGAO/NewKAONs/KAON496.pdf

    Function to estimate the full turbulence profile (ground layer + free
    atmosphere) of a particular night for which MASS and DIMM data was 
    recorded.

    Inputs:
    ------------
    dimm      : float
        Total seeing (arcsec) from DIMM data file

    mass_wts  : 7-entry 1D array, floats
        c_l weights from MASS data file. Response from 
        Mark Chun verifies that MASS file entries are indeed Cn^2*delta_h,
        as opposed to pure Cn^2 values. Cn^2*delta_h = c_l. NOTE: the
        last element of this array is the MASS seeing (not a weight),
        so it is not used in the calculation

    date      : string
        Date of MASS/DIMM data; used only for plot filename and labels

    plot      : boolean
        Option to plot turbulence profile after calculation and save to
        current working directory

    wvl       : float, default = 500 nm
        Wavelength involved in calculation

    normalize : boolean, default = True
        Option to normalize calculated turbulence profile

    Outputs:
    ------------
    wts       : 7-entry 1D array, floats
        Full turbulence profile with estimated ground layer weight

    r0        : float
        Fried parameter (meters)

    By Brooke DiGia
    """
    # Calculate 0th order turbulence moment
    r0 = fried(dimm, wvl)
    mu0 = 0.06 * ( wvl*1e-9 )**2.0 * r0**(-5.0/3.0)

    # Strip off MASS seeing
    mass_wts = mass_wts[:-1]
    c0 = mu0 - np.sum(mass_wts)
    cls = np.zeros(len(mass_wts) + 1)
    if normalize:
        to_normalize = np.insert(mass_wts, [0], c0)
        tot = np.sum(to_normalize)

        # Re-normalize turbulence weights with new ground layer entry
        cls = to_normalize/tot
    else:
        cls = np.insert(mass_wts, [0], c0)

    if plot:
        hts = np.array([0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
        plt.plot(hts, cls, "ro")
        plt.title("Turbulence profile %s" % date)
        plt.ylabel(r"c_{l} coefficients")
        plt.xlabel("Height above telescope (meters)")
        plt.show()
        plt.savefig("turb_profile_%s" % date)

    return r0, cls

def fetch_mass_dimm_cfht_loc(dimm_in_hrs, mass_in_hrs, 
                             cfht_in_hrs, timestamp):
    """
    Function to fetch the MASS/DIMM/CFHT data that is closest to 
    the input timestamp.

    Inputs:
    ------------
    dimm_in_hrs : float, array with length of data file
        DIMM time data expressed in hours

    mass_in_hrs : float, array with length of data file
        MASS time data expressed in hours

    cfht_in_hrs : float, variable length
        CFHT time data expressed in hours

    timestamp   : float
        Time for which to find the closest corresponding 
        MASS/DIMM/CFHT data

    Outputs:
    ------------
    dimm_idx    : int
        Index where closest DIMM data lies in DIMM file

    mass_idx    : int
        Index where closest MASS data lies in MASS file

    cfht_idx    : int
        Index where closest CFHT data lies in input CFHT subset

    By Brooke DiGia
    """
    dimm_time_diff = abs(dimm_in_hrs - timestamp)
    mass_time_diff = abs(mass_in_hrs - timestamp)
    cfht_time_diff = abs(cfht_in_hrs - timestamp)
    dimm_closest_idx = dimm_time_diff.argmin()
    mass_closest_idx = mass_time_diff.argmin()
    cfht_closest_idx = cfht_time_diff.argmin()

    if dimm_time_diff[dimm_closest_idx] > 1.0:
        print("Could not locate DIMM data close to ", timestamp)
    
    if mass_time_diff[mass_closest_idx] > 1.0:
        print("Could not locate MASS data close to ", timestamp)

    if cfht_time_diff[cfht_closest_idx] > 1.0:
        print("Could not locate CFHT data close to ", timestamp)

    return dimm_closest_idx, mass_closest_idx, cfht_closest_idx

def remove_keywords(file, *args):
    """
    Function to remove XSTREHL and YSTREHL keywords from input FITS 
    file headers. E.g. to remove these keywords from the on-sky 
    NIRC2 GC FITS files so that KAI's calc_strehl routine (modified 
    MAOS version above) does not use these XSTREHL and YSTREHL 
    coordinates for source coordinates when input FITS files are 
    PSFs (which by definition are centered on the Strehl source).

    Inputs:
    -----------
    file       : string
        Path to FITS file to be edited

    *args : string(s)
        Keyword(s) to be removed from input FITS file header

    Outputs:
    -----------
    None, input FITS file is revised in directory where it lives
    
    By Brooke DiGia
    """
    # Load header from input FITS file
    with fits.open(file) as fits_file:
        hdu = fits_file[0]

        # Remove all input keywords
        for word in args:
            print("Removing keyword '%s' from %s..." % (word, file))
            del hdu.header[word]

        # Overwrite original input file with desired changes
        hdu.writeto(file, overwrite=True)
            
    return

def estimate_on_sky_conditions(file, saveto, plot=False):
    """
    Function to take in the path to an on-sky .fits file (containing a PSF)
    and return an estimation of the atmospheric conditions present at the
    time of on-sky observation. The on-sky filename date and FITS header
    exposure start/stop times are recorded in UT. The MASS and DIMM contents
    are recorded in local Hawaiian time (HST), so this function will convert
    the MASS/DIMM data to UT. Likewise, the CFHT meteogram data is recorded
    in HST, so that data will be converted as well. The PHTO Hilo data is
    taken only twice per day, so this data does not need to be searched
    for the entry closest to the FITS exposure time. Instead, we search
    the PHTO data for the entries closest to the telescope heights
    in the MAOS atm config file.

    Inputs:
    ------------
    file             : string
        Path to on-sky FITS file

    saveto           : string
        Path of location to save MASS/DIMM data files

    plot             : boolean, default = False
        Option to plot turbulence profile and save to current working 
        directory

    Outputs:
    ------------
    avg_r0           : float
        Averaged Fried parameter in meters

    full_turb        : array, len=7, dtype=float
        Full turbulence profile (ground layer + free atmosphere)

    wind_spd_profile : array, len=7, dtype=float
        Full wind speed profile

    wind_dir_profile : array, len=7, dtype=float
        Full wind direction profile
    
    By Brooke DiGia
    """
    print("NOTE: Results for MAOS configuration files marked with ***\n")
    with fits.open(file) as fits_file:
        hdu = fits_file[0]
        psf = hdu.data
        hdr = hdu.header
        
        # Date of observation, parsed into year, month, day (UT)
        date = hdr["DATE"]
        year = date[:4]
        month = date[5:7]
        day = date[8:10]
        date_for_massdimm = year + month + day
        
        # Pull MASS and DIMM files corresponding to date of observation
        # if they are not already present in save_dir directory
        dimmdat = date_for_massdimm + ".dimm.dat"
        masspro = date_for_massdimm + ".masspro.dat"
        url_root = "http://mkwc.ifa.hawaii.edu/current/seeing/"
        url = url_root + "dimm/" + dimmdat
        if not os.path.exists(saveto + dimmdat):
            try:
                # Pull and save DIMM file
                urllib.request.urlretrieve(url, saveto + dimmdat)
                print(f"{dimmdat} saved to {saveto}")
            except Exception as error:
                print(f"Error while downloading {dimmdat} from {url}:", 
                      type(error).__name__, error)
                return
        else:
            print(f"{dimmdat} exists in directory {saveto}, not downloading.")

        # Reset url
        url = url_root + "masspro/" + masspro
        if not os.path.exists(saveto + masspro):
            try:
                # Pull and save MASS file
                urllib.request.urlretrieve(url, saveto + masspro)
                print(f"{masspro} saved to {saveto}")
            except Exception:
                print(f"Error while downloading {masspro} from {url}:", 
                      type(error).__name__, error)
                return
        else:
            print(f"{masspro} exists in directory {saveto}, not downloading.")

        # Pull CFHT data based on year of observation date
        cfht_url = "http://mkwc.ifa.hawaii.edu/archive/wx/cfht/cfht-wx.%s.dat" % year
        cfht = "cfht-wx.%s.dat" % year
        if not os.path.exists(saveto + cfht):
            try:
                urllib.request.urlretrieve(cfht_url, saveto + cfht)
                print(f"{cfht} saved to {saveto}")
            except Exception as error:
                print(f"Error while downloading {cfht} from {cfht_url}:",
                      type(error).__name__, error)
                return
        else:
            print(f"{cfht} exists in directory {saveto}, not downloading.")

        # Pull PHTO station data based on year of observation date
        phto_url = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR=" + year + "&MONTH=" + month + "&FROM=" + day + "00&TO=" + day + "00&STNM=91285"

        phto = "phto.%s.dat" % date_for_massdimm
        phto_clean = "phto.%s_cleaned.dat" % date_for_massdimm
        if not os.path.exists(saveto + phto):
            try:
                urllib.request.urlretrieve(phto_url, saveto + phto)
                with open(saveto + phto) as html:
                    soup = BeautifulSoup(html, 'html.parser')
                    to_save = soup.find_all('pre')[0]
                    for lines in to_save:
                        line = str(lines.text)
                        with open(saveto + phto_clean, 'w') as f:
                            f.write(line)
                print(f"{phto} saved to {saveto}")
            except Exception as error:
                print(f"Error while downloading {phto} from {phto_url}:",
                      type(error).__name__, error)
                return
        else:
            print(f"{phto} exists in directory {saveto}, not downloading.")

        # Exposure time on this date, parsed into hour, minute, second (UT)
        expstart = hdr["EXPSTART"]
        expstop = hdr["EXPSTOP"]
        print("\nDate of observation is  %s (UT)" % date)
        print("Exposure time is\t%s to %s (UT)\n" % (expstart, expstop))
        expstart_hr = float(expstart[:2])
        expstart_min = float(expstart[3:5])
        expstart_sec = float(expstart[6:8])
        expstop_hr = float(expstop[:2])
        expstop_min = float(expstop[3:5])
        expstop_sec = float(expstop[6:8])
        
        # Load in DIMM data and parse into individual arrays
        dimm_table = read_csv(saveto + dimmdat, delim_whitespace=True, names=\
                             ['year', 'month', 'day', 'hour', 'minute', 'second', \
                             'seeing'])
        dimm_yr = np.array(dimm_table['year'])
        dimm_mon = np.array(dimm_table['month'])
        dimm_day = np.array(dimm_table['day'])
        dimm_hr = np.array(dimm_table['hour'])
        dimm_min = np.array(dimm_table['minute'])
        dimm_sec = np.array(dimm_table['second'])
        dimm_seeing = np.array(dimm_table['seeing'])

        # Convert DIMM data from local Hawaiian time (HST) to UT to match FITS
        # date of observation and exposure time
        dimm_hr += 10
        idx = np.where(dimm_hr >= 24)[0]
        dimm_day[idx] += 1
        dimm_hr[idx] -= 24

        # Create array with DIMM time data expressed in only hours
        dimm_time_in_hrs = np.add(dimm_hr, np.divide(dimm_min, 60.0), 
                                  np.divide(dimm_sec, 3600.0))

        # Load in MASS data and parse into individual arrays
        mass_table = read_csv(saveto + masspro, delim_whitespace=True, names=\
                             ['year', 'month', 'day', 'hour', 'minute', 'second', \
                              'cn2dh_05', 'cn2dh_1', 'cn2dh_2', 'cn2dh_4', \
                              'cn2dh_8', 'cn2dh_16', 'seeing'])
        mass_yr = np.array(mass_table['year'])
        mass_mon = np.array(mass_table['month'])
        mass_day = np.array(mass_table['day'])
        mass_hr = np.array(mass_table['hour'])
        mass_min = np.array(mass_table['minute'])
        mass_sec = np.array(mass_table['second'])
        mass_cn2dh05 = np.array(mass_table['cn2dh_05'])
        mass_cn2dh1 = np.array(mass_table['cn2dh_1'])
        mass_cn2dh2 = np.array(mass_table['cn2dh_2'])
        mass_cn2dh4 = np.array(mass_table['cn2dh_4'])
        mass_cn2dh8 = np.array(mass_table['cn2dh_8'])
        mass_cn2dh16 = np.array(mass_table['cn2dh_16'])
        mass_seeing = np.array(mass_table['seeing'])

        # Convert MASS data from HST to UT
        mass_hr += 10
        idx = np.where(mass_hr >= 24)[0]
        mass_day[idx] += 1
        mass_hr[idx] -= 24

        # Create array with MASS time data in hours
        mass_time_in_hrs = np.add(mass_hr, np.divide(mass_min, 60.0),
                                  np.divide(mass_sec, 3600.0))

        # Load in CFHT data and parse into individual arrays
        cfht_table = read_csv(saveto + cfht, delim_whitespace=True, usecols=\
                             [0, 1, 2, 3, 4, 5, 6], names=['year', 'month', \
                                                           'day', 'hour', 'minute', \
                                                           'wdspd', 'wddir'])
        cfht_yr = np.array(cfht_table['year'])
        cfht_mon = np.array(cfht_table['month'])
        cfht_day = np.array(cfht_table['day'])
        cfht_hr = np.array(cfht_table['hour'])
        cfht_min = np.array(cfht_table['minute'])
        # Convert wind speed in knots to m/s. 1 knot = 1852 m/hr
        cfht_wdspd = np.array(cfht_table['wdspd']) * ( 1852.0 / (60.0 * 60.0) )
        # Wind direction in degrees
        cfht_wddir = np.array(cfht_table['wddir'])

        # Convert CFHT data from HST to UT
        cfht_hr += 10
        idx = np.where(cfht_hr >= 24)[0]
        cfht_day[idx] += 1
        cfht_hr[idx] -= 24

        # Extract CFHT data that corresponds to date of observation
        # (CFHT data file contains data for the entire year, which we
        # (do not need for one night of observation)
        cfht_i = np.where( (cfht_mon == int(month)) & 
                           (cfht_day == int(day)) )
        cfht_yr = cfht_yr[cfht_i]
        cfht_mon = cfht_mon[cfht_i]
        cfht_day = cfht_day[cfht_i]
        cfht_hr = cfht_hr[cfht_i]
        cfht_min = cfht_min[cfht_i]
        cfht_wdspd = cfht_wdspd[cfht_i]
        cfht_wddir = cfht_wddir[cfht_i]

        # Create array with CFHT time data in hours
        cfht_time_in_hrs = np.add(cfht_hr, np.divide(cfht_min, 60.0))

        # Load in PHTO data and parse into individual arrays
        phto_table = read_csv(saveto + phto_clean, delim_whitespace=True, 
                              usecols=[1,6,7], skiprows=[0,1,2,3,4], 
                              names=['hght', 'drct', 'sknt'], 
                              skipfooter=1, engine='python')
        phto_hghts = np.asarray(phto_table['hght'], dtype=float)
        phto_wddir = np.array(phto_table['drct'])
        phto_wdspd = np.array(phto_table['sknt']) * ( 1852.0 / (60.0 * 60.0) )

        # PHTO heights are relative to sea level, but MAOS atm.ht is
        # height above telescope, so we need to subtract the height
        # of Keck (4145 m) from phto_hghts to convert to height
        # above telescope
        phto_hghts -= 4145.0
        # Discard entries with negative heights
        idx = np.where(phto_hghts > 0.0)[0]
        phto_hghts = phto_hghts[idx]
        # Find PHTO data closest to the following heights (meters); ground layer (0.0 m) calculated with CFHT data
        heights = np.array([500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
        phto_indices = np.zeros(len(heights), dtype=int)
        for i in range(len(heights)):
            phto_hght_diff = abs(phto_hghts - heights[i])            
            phto_indices[i] = phto_hght_diff.argmin()

        # Fetch MASS/DIMM/CFHT data closest to exposure time
        expstart_float = expstart_hr + (expstart_min/60.0) + (expstart_sec/3600.0)
        expstop_float = expstop_hr + (expstop_min/60.0) + (expstop_sec/3600.0)
        i_dimm_start, i_mass_start, i_cfht_start = fetch_mass_dimm_cfht_loc(dimm_time_in_hrs, 
                                                                            mass_time_in_hrs, 
                                                                            cfht_time_in_hrs,
                                                                            expstart_float)
        i_dimm_stop, i_mass_stop, i_cfht_stop = fetch_mass_dimm_cfht_loc(dimm_time_in_hrs,
                                                                         mass_time_in_hrs, 
                                                                         cfht_time_in_hrs,
                                                                         expstop_float)
        closest_dimm_start = dimm_seeing[i_dimm_start]
        closest_dimm_stop = dimm_seeing[i_dimm_stop]
        avg_dimm = (closest_dimm_start + closest_dimm_stop) / 2.0
        dimm_date_start = str(dimm_yr[i_dimm_start]) \
                          + ":" + str(dimm_mon[i_dimm_start]) \
                          + ":" + str(dimm_day[i_dimm_start]) \
                          + ":" + str(dimm_hr[i_dimm_start]) \
                          + ":" + str(dimm_min[i_dimm_start]) \
                          + ":" + str(dimm_sec[i_dimm_start])
        dimm_date_stop = str(dimm_yr[i_dimm_stop]) \
                         + ":" + str(dimm_mon[i_dimm_stop]) \
                         + ":" + str(dimm_day[i_dimm_stop]) \
                         + ":" + str(dimm_hr[i_dimm_stop]) \
                         + ":" + str(dimm_min[i_dimm_stop]) \
                         + ":" + str(dimm_sec[i_dimm_stop])
        print("Closest DIMM data to beginning of exposure:\t%0.4f\tat %s" % 
              (closest_dimm_start, dimm_date_start))
        print("Closest DIMM data to end of exposure:\t\t%0.4f\tat %s" % 
              (closest_dimm_stop, dimm_date_stop))
        print("Average DIMM over exposure:\t\t\t%0.4f\n" % avg_dimm)
        
        # Some output formatting
        np.set_printoptions(precision=3)

        mass_profile_start = [mass_cn2dh05[i_mass_start], mass_cn2dh1[i_mass_start],
                              mass_cn2dh2[i_mass_start], mass_cn2dh4[i_mass_start],
                              mass_cn2dh8[i_mass_start], mass_cn2dh16[i_mass_start],
                              mass_seeing[i_mass_start]]
        mass_profile_stop = [mass_cn2dh05[i_mass_stop], mass_cn2dh1[i_mass_stop],
                             mass_cn2dh2[i_mass_stop], mass_cn2dh4[i_mass_stop],
                             mass_cn2dh8[i_mass_stop], mass_cn2dh16[i_mass_stop],
                             mass_seeing[i_mass_stop]]
        mass_date_start = str(mass_yr[i_mass_start]) \
                          + ":" + str(mass_mon[i_mass_start]) \
                          + ":" + str(mass_day[i_mass_start]) \
                          + ":" + str(mass_hr[i_mass_start]) \
                          + ":" + str(mass_min[i_mass_start]) \
                          + ":" + str(mass_sec[i_mass_start])
        mass_date_stop = str(mass_yr[i_mass_stop]) \
                         + ":" + str(mass_mon[i_mass_stop]) \
                         + ":" + str(mass_day[i_mass_stop]) \
                         + ":" + str(mass_hr[i_mass_stop]) \
                         + ":" + str(mass_min[i_mass_stop]) \
                         + ":" + str(mass_sec[i_mass_stop])
        print("Closest MASS data to beginning of exposure: ", 
              np.array(mass_profile_start), "at ", mass_date_start)
        print("Closest MASS data to end of exposure:\t    ", 
              np.array(mass_profile_stop), "at ", mass_date_stop)

        cfht_date_start = str(cfht_yr[i_cfht_start]) \
                          + ":" + str(cfht_mon[i_cfht_start]) \
                          + ":" + str(cfht_day[i_cfht_start]) \
                          + ":" + str(cfht_hr[i_cfht_start]) \
                          + ":" + str(cfht_min[i_cfht_start])
        cfht_date_stop = str(cfht_yr[i_cfht_stop]) \
                         + ":" + str(cfht_mon[i_cfht_stop]) \
                         + ":" + str(cfht_day[i_cfht_stop]) \
                         + ":" + str(cfht_hr[i_cfht_stop]) \
                         + ":" + str(cfht_min[i_cfht_stop])
        
        # Estimate turbulence for beginning and end of exposure
        r0_start, start_turb = estimate_turbulence(closest_dimm_start, 
                                                   mass_profile_start,
                                                   date_for_massdimm, 
                                                   plot)
        r0_end, end_turb = estimate_turbulence(closest_dimm_stop, 
                                               mass_profile_stop,
                                               date_for_massdimm,
                                               plot)
        # Average these two turbulence profiles together (we average due to potential
        # noise involved in the MASS/DIMM measurements)
        avg_turb = np.average((np.array(start_turb), np.array(end_turb)), axis=0)
        avg_r0 = (r0_start + r0_end) / 2.0
        avg_mass_seeing = (mass_profile_start[-1] + mass_profile_stop[-1]) / 2.0
        print("\nMASS seeing (arcsec): ", avg_mass_seeing)
        print("*** Full turbulence profile:\t\t    ", avg_turb)
        print("*** Fried parameter (m):\t\t\t%0.6f\n" % avg_r0)

        # Average CFHT data across exposure to calculate ground layer wind speed
        # and direction
        cfht_wdspd_start = cfht_wdspd[i_cfht_start]
        cfht_wddir_start = cfht_wddir[i_cfht_start]
        cfht_wdspd_stop = cfht_wdspd[i_cfht_stop]
        cfht_wddir_stop = cfht_wddir[i_cfht_stop]
        print("Closest CFHT data to beginning of exposure: %0.4f (m/s) | %0.4f (deg) at %s" % 
              (cfht_wdspd_start, cfht_wddir_start, cfht_date_start))
        print("Closest CFHT data to end of exposure:\t    %0.4f (m/s) | %0.4f (deg) at %s\n" %
              (cfht_wdspd_stop, cfht_wddir_stop, cfht_date_stop))

        avg_grnd_wdspd = ( cfht_wdspd_start + cfht_wdspd_stop ) / 2.0
        avg_grnd_wddir = ( cfht_wddir_start + cfht_wddir_stop ) / 2.0
        free_atm_wdspd = phto_wdspd[phto_indices]
        free_atm_wddir = phto_wddir[phto_indices]
        wind_spd_profile = np.concatenate( (np.array([avg_grnd_wdspd]), 
                                                     free_atm_wdspd) )
        wind_dir_profile = np.concatenate( (np.array([avg_grnd_wddir]), 
                                                     free_atm_wddir) )
        print("Free atm wind speed/direction profiles taken at these heights:", 
              phto_hghts[phto_indices])
        print("*** Full wind speed profile (m/s): ", wind_spd_profile)
        print("*** Full wind direction profile (deg): ", wind_dir_profile)
        
    return avg_r0, avg_turb, wind_spd_profile, wind_dir_profile

def maos_windshake_grid(amps, on_sky, thres=0.05):
    """
    Function to run multiple MAOS simulations for a grid search of various
    windshake amplitudes (mas).

    Inputs:
    ------------
    amps   : array, variable length, dtype=float
        Total vibration-jitter amplitudes to be input into make_keck_vib_psd()

    on_sky : 2D-array, variable length, dtype=mixed (str + float)
        Array containing the frame names, their locations, and metrics for on-sky
        data. The frame name is the name ONLY (no path or file suffix), while the
        location must be a FULL path (not relative). An example row of this array:

        ['c0103', '/u/bdigia/work/ao/single_psfs/good_run_psfs/', 0.303,
         53.49, 369.5]

    thres  : float, default = 0.05
	Optional threshold argument for 'passing' criteria in grid search
        
    Outputs:
    ------------
    best   : array, variable length, 8-element tuples of floats
        Array of best combinations of r0 and l0 and their corresponding
        strehls, fwhms, and rmswfes

    Also displays simulation results in terminal and writes output metrics to
    text file defined below as "metrics_file" via the calc_strehl function

    By Brooke DiGia
    """
    base_root = "/u/bdigia/work/ao/keck/maos/keck/my_base/"
    best = []

    # Convert on_sky array to numpy array in case user
    # did not input it as such
    on_sky = np.array(on_sky)

    for amp in amps:
    	for i in range(on_sky.shape[0]):
            # Get atmospheric conditions for current on_sky frame
            fried, turbpro, windspd, winddrct = estimate_on_sky_conditions(on_sky[i][1]+on_sky[i][0]+"_psf.fits", on_sky[i][1])
            
            # Make new PSD based on input total jitter amplitude
            psd_file = keck_utils.make_keck_vib_psd(amp)
        
            # Set MAOS command
            folder = f"A_keck_scao_lgs_gc_ws={amp}mas_{on_sky[i][0]}"
            maos_cmd = f"""maos -o {folder} -c A_keck_scao_lgs_gc.conf plot.all=1 plot.setup=1 sim.wspsd={psd_file} atm.r0z={fried} atm.wt={turbpro} atm.ws={windspd} atm.wddeg={winddrct} -O"""

            cwd = os.getcwd()
            # Must be in MAOS simulation directory to run successfully
            if cwd != base_root:
                print("Current working directory (CWD) is %s" % cwd)
                print("Moving CWD to MAOS simulation directory...\n")
                os.chdir(base_root)

            os.system(maos_cmd)

    # After all simulations are run, fetch and display results
    for amp in amps:
        for i in range(on_sky.shape[0]):
            folder = f"A_keck_scao_lgs_gc_ws={amp}mas_{on_sky[i][0]}"
            metrics_file = base_root + folder + "_sim_results.txt"
            sim_dir = base_root + folder + "/"
            print(f"\n\n **** Total Jitter = {amp} mas ****")
            strehl_array, fwhm_array, rmswfe_array = calc_strehl(sim_dir, metrics_file, 
                                                                 skysub=False, apersize=0.3)

            # Currently comparing to on_sky only at 2.12 microns
            delta_strehl = abs(float(on_sky[i][2]) - strehl_array[-1])
            delta_fwhm = abs(float(on_sky[i][3]) - fwhm_array[-1])
            delta_rmswfe = abs(float(on_sky[i][4]) - rmswfe_array[-1])
            if (delta_strehl <= thres) & (delta_fwhm <= thres):
                tuple = (amp, strehl_array[-1], delta_strehl, fwhm_array[-1], delta_fwhm, 
                         rmswfe_array[-1], delta_rmswfe)
                best.append(tuple)

    # Sort resultant 'best' tuples based on delta Strehl values
    best.sort(key=lambda tup: tup[2])
    print("\n\n***** Best combinations *****")
    print("Jitter amplitude (mas) | Strehl | Delta Strehl | FWHM (mas) | Delta FWHM (mas) | RMS WFE (nm) | Delta RMS WFE (nm)")
    print(best)
    return best

def maos_phase_screen_grid(r0s, l0s, on_sky, thres=0.05):
    """
    Function to test multiple NCPA r0 and l0 parameter values, in comparison
    to an on-sky Strehl quantity (currently hard-coded below). The best 
    (r0, l0) combinations are determined via the input threshold.

    Inputs:
    ----------
    r0s    : array, variable length, dtype=float
        Array of NCPA surf r0 values for which to run MAOS base Keck simulation

    l0s    : array, variable length, dtype=float
        Array of NCPA surf l0 values for which to run MAOS base Keck simulation

    on_sky : 2D-array, variable length, dtype=mixed (str + float)
        Array containing the frame names, their locations, and metrics for on-sky
        data. The frame name is the name ONLY (no path or file suffix), while the 
        location must be a FULL path (not relative). An example row of this array:

        ['c0103', '/u/bdigia/work/ao/single_psfs/good_run_psfs/', 0.303,
         53.49, 369.5]

    thres  : float, default = 0.05
        Threshold for determining if a phase screen is "best" for output

    Outputs:
    ----------
    best  : array, variable length, 8-element tuples of floats
        Array of best combinations of r0 and l0 and their corresponding
        strehls, fwhms, and rmswfes

    Also displays simulation results in terminal and writes output metrics to
    text file defined below as "metrics_file" via the calc_strehl function

    By Brooke DiGia
    """
    base_root = "/u/bdigia/work/ao/keck/maos/keck/my_base/"
    best = []

    # Convert on_sky array to numpy array in case user
    # did not input it as such
    on_sky = np.array(on_sky)

    for r0 in r0s:
        for l0 in l0s:
            for i in range(on_sky.shape[0]):
                # Get atmospheric conditions for current on_sky frame
                fried, turbpro, windspd, winddrct = estimate_on_sky_conditions(on_sky[i][1]+on_sky[i][0]+"_psf.fits", on_sky[i][1])

                # Set MAOS command based on current r0 and l0
                maos_cmd = f"""maos -o A_keck_scao_lgs_gc_r0={r0}_l0={l0}_{on_sky[i][0]} -c A_keck_scao_lgs_gc.conf plot.all=1 plot.setup=1 surf=["Keck_ncpa_rmswfe130nm.fits", "'r0={r0};l0={l0};ht=40000;slope=-2; SURFWFS=1; SURFEVL=1; seed=10;'"] atm.r0z={fried} atm.wt={turbpro} atm.ws={windspd} atm.wddeg={winddrct} -O"""

                cwd = os.getcwd()
                # Must be in MAOS simulation directory to run successfully
                if cwd != base_root:
                    print("Current working directory (CWD) is %s" % cwd)
                    print("Moving CWD to MAOS simulation directory...\n")
                    os.chdir(base_root)

                try:
                    os.system(maos_cmd)
                except Exception as error:
                    print("Error running MAOS: ", error)
                    return

    # After all simulations are run, fetch and display results
    for r0 in r0s:
        for l0 in l0s:
            for i in range(on_sky.shape[0]):
                folder = f"A_keck_scao_lgs_gc_r0={r0}_l0={l0}_{on_sky[i][0]}"
                metrics_file = base_root + folder + "_sim_results.txt"
                sim_dir = base_root + folder + "/"
                print(f"\n\n **** r0 = {r0} | l0 = {l0} ****")
                strehl_array, fwhm_array, rmswfe_array = calc_strehl(sim_dir, 
                                                                     metrics_file, 
                                                                     skysub=False, 
                                                                     apersize=0.3)
                # Currently comparing to on_sky only at 2.12 microns
                delta_strehl = abs(float(on_sky[i][2]) - strehl_array[-1])
                delta_fwhm = abs(float(on_sky[i][3]) - fwhm_array[-1])
                delta_rmswfe = abs(float(on_sky[i][4]) - rmswfe_array[-1])
                if (delta_strehl <= thres) & (delta_fwhm <= thres):
                    tuple = (r0, l0, strehl_array[-1], delta_strehl, fwhm_array[-1], 
                             delta_fwhm, rmswfe_array[-1], delta_rmswfe)
                    best.append(tuple)

    # Sort resultant 'best' tuples based on delta Strehl values
    best.sort(key=lambda tup: tup[3])
    print("\n\n***** Best combinations *****")
    print("r0 | l0 | Strehl | Delta Strehl | FWHM (mas) | Delta FWHM (mas) | RMS WFE (nm) | Delta RMS WFE (nm)")
    print(best)
    return best

def make_strehl_plot(maos_txts, saveto, file_suffix=".pdf"):
    """
    Function to take an array of MAOS simulation results
    and plot the Strehl results at 2.12 microns versus the 
    corresponding on-sky Strehl results (available at only
    2.12 microns). 

    Inputs:
    ----------
    maos_txts   : array, variable length, dtype=string
        Array of MAOS simulation Strehl calculator output
        text files containing results to plot.

    saveto      : string
        Location to save plot images

    file_suffix : string, default=.pdf
        File type for saving plot (PDF or PNG)

    Outputs:
    ----------
    None, plot image (PDF or PNG) saved to desired output
    folder

    By Brooke DiGia
    """
    filename = "MAOS_Strehl_Plot"
    on_sky = np.zeros(len(maos_txts))
    maos = np.zeros(len(maos_txts))
    on_sky_root = "/u/bdigia/work/ao/single_psfs/good_run_psfs/"

    for i in range(len(maos_txts)):
        frame = maos_txts[i][-9:-4]
        on_sky = calc_strehl_on_sky([on_sky_root + frame + "_psf.fits"], 
                                     on_sky_root + frame + "_GC_Strehl.txt", 
                                     apersize=0.3)[0]
        # Read in MAOS result at 2.12 microns only
        table = read_csv(maos_txts[i], delim_whitespace=True, 
                         skiprows=[0,1,2,3,4,5], usecols=[1], names=['strehl'])
        print("TABLE")
        print(table['strehl'])

    plt.figure()
    plt.xlabel("MAOS Strehl (calculated via PAARTI)")
    plt.ylabel("On-Sky GC Strehl")
    plt.plot(maos, on_sky, "bo")
    plt.savefig(saveto + filename + file_suffix)
    return

"""
The following *_on_sky() functons are copied from the KAI repository, linked
above in function headers, for use on on-sky PSF images. This is to keep the
KAI pipeline unchanged in its own repository.
"""

def calc_strehl_on_sky(file_list, out_file, apersize=0.3, 
                       instrument=None,
                       skysub=False):
    """
    Calculate the Strehl, FWHM, and RMS WFE for each image in a
    list of files. The output is stored into the specified <out_file>
    text file. The FWHM (and Strehl) is calculated over the specified
    aperture size using a 2D gaussian fit. The Strehl is estimated by
    taking the max pixel flux / wide-aperture flux and normalizing
    by the same on a diffraction-limited image. Note that the diffraction
    limited image comes from an external file.

    The diffraction limited images come with the pipeline. For Keck, they are
    all obtained empirically using the NIRC2 camera and filters and they
    are sampled at 0.009952 arcsec / pixel. We will resample them as necessary.
    We will play fast and loose with them and use them for both NIRC2 and OSIRIS.
    They will be resampled as needed.

    Inputs
    ----------
    file_list : list or array
        The list of the file names.

    out_file : str
        The name of the output text file.

    aper_size : float (def = 0.3 arcsec)
        The aperture size over which to calculate the Strehl and FWHM.

    skysub    : boolean (def = False)
        Option to perform sky subtraction on input PSF
    """
    from kai import instruments
    if instrument is None:
        instruments.default_inst

    # Setup the output file and format.
    _out = open(out_file, 'w')

    fmt_hdr = '{img:<30s} {strehl:>7s} {rms:>7s} {fwhm:>7s}  {mjd:>10s}\n'
    fmt_dat = '{img:<30s} {strehl:7.3f} {rms:7.1f} {fwhm:7.2f}  {mjd:10.4f}\n'

    _out.write(fmt_hdr.format(img='#Filename', strehl='Strehl', rms='RMSwfe', 
                              fwhm='FWHM', mjd='MJD'))
    _out.write(fmt_hdr.format(img='#()', strehl='()', rms='(nm)', 
                              fwhm='(mas)', mjd='(UT)'))

    # Find the root directory where the calibration files live.
    base_path = "/u/bdigia/code/python/KAI/kai"
    cal_dir = base_path + '/data/diffrac_lim_img/' + instrument.telescope + '/'

    # We are going to assume that everything in this list
    # has the same camera, filter, plate scale, etc.
    img0, hdr0 = fits.getdata(file_list[0], header=True)
    filt = instrument.get_filter_name(hdr0)
    scale = instrument.get_plate_scale(hdr0)
    wavelength = instrument.get_central_wavelength(hdr0)
    print("Filter = %s | Scale (arcsec/px) = %f | Wavelength (microns) = %f " % 
          (filt, scale, wavelength))

    # Get the diffraction limited image for this filter.
    dl_img_file = cal_dir + filt.lower() + '.fits'
    dl_img, dl_hdr = fits.getdata(dl_img_file, header=True)

    # Get the DL image scale and re-scale it to match the science iamge.
    if 'Keck' in instrument.telescope:
        scale_dl = 0.009952  # Hard-coded
    else:
        scale_dl = dl_img['PIXSCALE']
    rescale = scale_dl / scale

    if rescale != 1:
        dl_img = scipy.ndimage.zoom(dl_img, rescale, order=3)

    # Pick appropriate radii for extraction.
    # The diffraction limited resolution in pixels.
    dl_res_in_pix = 0.25 * wavelength / (instrument.telescope_diam * scale)
    radius = int(np.ceil(apersize / scale))
    if radius < 3:
        radius = 3
    
    # Perform some wide-aperture photometry on the diffraction-limited image.
    # We will normalize our Strehl by this value. We will do the same on the
    # data later on.
    peak_coords_dl = np.unravel_index(np.argmax(dl_img, axis=None), dl_img.shape)
    plt.figure()
    plt.imshow(dl_img)
    plt.savefig("/u/bdigia/work/ao/single_psfs/good_run_psfs/test_on_sky_dl_plot.pdf")
    # Calculate the peak flux ratio
    try:
        dl_peak_flux_ratio = calc_peak_flux_ratio_on_sky(dl_img, peak_coords_dl, 
                                                         radius, skysub)
        print("dl_peak_flux_ratio:", dl_peak_flux_ratio)
        # For each image, get the strehl, FWHM, RMS WFE, MJD, etc. and write to an
        # output file.
        strehls = []
        for ii in range(len(file_list)):
            strehl, fwhm, rmswfe = calc_strehl_single_on_sky(file_list[ii], radius, 
                                                             dl_peak_flux_ratio, 
                                                             instrument=instrument, skysub=skysub)
            strehls.append(strehl)
            mjd = fits.getval(file_list[ii], instrument.hdr_keys['mjd'])
            dirname, filename = os.path.split(file_list[ii])

            _out.write(fmt_dat.format(img=filename, strehl=strehl, rms=rmswfe, 
                                      fwhm=fwhm, mjd=mjd))
            print(fmt_dat.format(img=filename, strehl=strehl, rms=rmswfe, 
                                 fwhm=fwhm, mjd=mjd))
        _out.close()
    except astropy.nddata.PartialOverlapError:
        print("astropy.nddata.PartialOverlapError, failing gracefully...")
        for ii in range(len(file_list)):
            _out.write(fmt_dat.format(img=filename, strehl=-1.0, rms=-1.0, 
                                      fwhm=-1.0, mjd=mjd))
            print(fmt_dat.format(img=filename, strehl=-1.0, rms=-1.0, 
                                 fwhm=-1.0, mjd=mjd))
        _out.close()
    return strehls

def calc_strehl_single_on_sky(img_file, radius, dl_peak_flux_ratio, 
                              skysub, instrument=None):


    from kai import instruments
    if instrument is None:
        instruments.default_inst    
    # Read in the image and header.
    img, hdr = fits.getdata(img_file, header=True)
    wavelength = instrument.get_central_wavelength(hdr) # microns
    scale = instrument.get_plate_scale(hdr)

    # Position of Strehl source
    coords = np.array([img.shape[0]/2.0, img.shape[1]/2.0])

    # Use Strehl source coordinates in the header, if available and recorded
    if 'XSTREHL' in hdr:
        print(hdr)
        coords = np.array([float(hdr['XSTREHL']),
                           float(hdr['YSTREHL'])])
        coords -= 1     # Coordinate were 1 based; but python is 0 based.
    
    # Calculate the FWHM using a 2D gaussian fit. We will just average the two.
    # To make this fit more robust, we will change our boxsize around, slowly
    # shrinking it until we get a reasonable value.

    # First estimate the DL FWHM in pixels. Use this to set the boxsize for
    # the FWHM estimation... note that this is NOT the aperture size specified
    # above which is only used for estimating the Strehl.
    dl_res_in_pix = 0.25 * wavelength / (instrument.telescope_diam * scale)
    fwhm_min = 0.9 * dl_res_in_pix
    fwhm_max = 100
    fwhm = 0.0
    fwhm_boxsize = int(np.ceil((4 * dl_res_in_pix)))
    if fwhm_boxsize < 3:
        fwhm_boxsize = 3
    pos_delta_max = 2*fwhm_min
    box_scale = 1.0
    iters = 0

    # Steadily increase the boxsize until we get a reasonable FWHM estimate.
    while ((fwhm < fwhm_min) or (fwhm > fwhm_max)) and (iters < 30):
        box_scale += iters * 0.1
        iters += 1
        g2d = fit_gaussian2d(img, coords, fwhm_boxsize*box_scale,
                             fwhm_min=0.8*fwhm_min, fwhm_max=fwhm_max,
                             pos_delta_max=pos_delta_max)
        sigma = (g2d.x_stddev_0.value + g2d.y_stddev_0.value) / 2.0
        fwhm = stddev_to_fwhm(sigma)

        print(img_file.split('/')[-1], iters, fwhm,
                  g2d.x_mean_0.value, g2d.y_mean_0.value, fwhm_boxsize*box_scale)

        # Update the coordinates if they are reasonable. 
        if ((np.abs(g2d.x_mean_0.value - coords[0]) < fwhm_boxsize) and
            (np.abs(g2d.y_mean_0.value - coords[1]) < fwhm_boxsize)):
            coords = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])

    # Convert to milli-arcseconds
    fwhm *= scale * 1e3

    # Calculate the peak flux ratio
    peak_flux_ratio = calc_peak_flux_ratio_on_sky(img, coords, radius, skysub)

    # Normalize by the same from the DL image to get the Strehl.
    strehl = peak_flux_ratio / dl_peak_flux_ratio
    print('peak flux ratio = ', peak_flux_ratio, ' dl peak flux ratio = ', dl_peak_flux_ratio)

    # Convert the Strehl to a RMS WFE using the Marechal approximation.
    rms_wfe = np.sqrt( -1.0 * np.log(strehl)) * wavelength * 1.0e3 / (2. * math.pi)
    
    # Check final values and fail gracefully.
    if ((strehl < 0) or (strehl > 1) or
        (fwhm > 500) or (fwhm < (fwhm_min * scale * 1e3))):
        strehl = -1.0
        fwhm = -1.0
        rms_wfe = -1.0

    fmt_dat = '{img:<30s} {strehl:7.3f} {rms:7.1f} {fwhm:7.2f} {xpos:6.1f} {ypos:6.1f}\n'
    print(fmt_dat.format(img=img_file, strehl=strehl, rms=rms_wfe, fwhm=fwhm, xpos=coords[0], ypos=coords[1]))
    
    return strehl, fwhm, rms_wfe

def calc_peak_flux_ratio_on_sky(img, coords, radius, skysub):
    """
    img : 2D numpy array
        The image on which to calculate the flux ratio of the peak to a 
        wide-aperture.

    coords : list or numpy array, length = 2
        The x and y position of the source.

    radius : int
        The radius, in pixels, of the wide-aperture. 

    skysub : boolean
        Option to perform sky subtraction.

    """
    # Determine the peak flux
    peak_coords = np.unravel_index(np.argmax(img.data, axis=None), 
                                             img.data.shape)
    peak_flux = img[peak_coords]
    
    # Calculate the Strehl by first finding the peak-pixel flux / wide-aperture flux.
    # Then normalize by the same thing from the reference DL image. 
    aper_sum = np.sum(img)

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

