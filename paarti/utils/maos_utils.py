
import numpy as np
import math
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
import astropy.units as u
import astropy
from paarti.psf_metrics import metrics
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
import glob
from scipy import stats, signal
import scipy, scipy.misc, scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import urllib
from pandas import read_csv
import pandas as pd
from kai import instruments
from bs4 import BeautifulSoup
import readbin # from MAOS
from pathlib import Path
from scipy.io import readsav
from scipy import signal
from matplotlib.ticker import FuncFormatter

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

def seeing_limit_spot_size(wvl:u.m, r0:u.m) -> u.arcsec:
    """
    Function to calculate the seeing-limited spot size
    based on wavelength and Fried parameter r0. 

    Seeing full width half maximum of the seeing disk formula:
    http://www.eso.org/gen-fac/pubs/astclim/papers/lz-thesis/node11.html

    Inputs:
    -------
    wvl     : float
        Wavelength in meters. For a wavefront sensor operating in a 
        certain band, use the wavelength on which the band is centered

    r0      : float
        Fried parameter in meters

    Outputs:
    --------
    theta   : float
        Seeing-limited spot size in arcseconds

    By Brooke DiGia
    """
    theta = (2.013*1.0e5) * (wvl/r0)
    return theta

def keck_nea_photons(m:float, wfs:str, r0:float, wfs_int_time:float=1.0/800.0, lbwfs_fwhm=None):
    """
    Calculate the number of photons, number of background photons,
    and noise equivalent angle for a natural guide star.

    Inputs:
    -------
    m              : float
        Magnitude of guide star
    wfs            : str
        Name of WFS to set camera properties
    r0             : float
        Fried parameter (m)
    - B.DiGia 12/13/2024: Fried parameter now required as an input in 
    estimating theta_beta (convolution with seeing-limited disk,
    which is determined by wavelength and r0)

    Optional Inputs:
    ----------------
    wfs_int_time   : float
        Integration time of the WFS in seconds

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

    SHWFS (HO, fast) is CCD39, gain measured to be 0.508 +/- 0.10 e-/ADU,
    according to KAON 387 by Marcos van Dam and Erik Johansson. This KAON
    is not available on the sharepoint - see Keck AO public in MULab drive.
    """
    # LGSWFS-OCAM2K  : KAPA and KAPA+HODM simulation setups
    # LGS-HODM-HOWFS : KAPA+HODM+HOWFS
    # wfs_list = ['LBWFS', 'LGSWFS', 'LGSWFS-OCAM2K', 'LGS-HODM-HOWFS', 
    #             'TRICK-H', 'TRICK-K', 'STRAP']

    # if wfs not in wfs_list:
    #     raise RuntimeError("keck_nea_photons: Invalid WFS.")
    
    # Keck telescope diameter (m)
    D = 10.949
    # Secondary obscuration diameter (m)
    Ds = 1.8
    
    # Parameter definitons:
    # wavelength : Guide star imaging wavelength
    # ps         : Pixel scale (arcsec/px)
    # sigma_e    : RMS detector read noise per pixel
    # theta_beta : Spot size on detector (rad)
    # pix_per_ap : Pixels per subaperture, for noise calculation
    
    if wfs == 'LBWFS':
        band = "R"
        band_wvl = 0.641e-6
        # wavelength = 0.641e-6

        # side length of square subaperture (m)
        side = 0.563 

        # KAON 265 " LBWFS has 16.7x16.7 pixels per subaperture at 0.148 ''/px "
        ps = 0.148 # previously 1.5
        
        # KAON 1303 has LBWFS readnoise as 5.82 e-, but KAON 245 has readnoise 3 e/pix
        sigma_e = 7.96 # for 2017 LBWFS replacement KAON 1303 pg 22
        
        # from KAON 1303 Table 16 - this table includes spot size measurements
        # from sacnning an AO single mode fiber source across the CCD-39 camera pixels
        # using the AO tip-tilt mirror.
        # B. DiGia 12/13/2024 - this spot size is intrinsic to the WFS and should be
        # convolved with the seeing-limited disk for the full WFS spot size
        theta_r0 = seeing_limit_spot_size(band_wvl, r0)
        if lbwfs_fwhm is not None:
            # If an LBWFS FWHM is provided, use that instead of the default 0.5''
            theta_beta = np.sqrt(theta_r0**2.0 + (lbwfs_fwhm/2.355)**2.0)
        else:
            theta_beta = np.sqrt(theta_r0**2.0 + 0.5**2.0)

        # Convert spot size to radians
        theta_beta *= ( math.pi/180.0 ) / ( 60.0*60.0 )
        
        # from KAON 1303 Table 7
        throughput = 0.03
        
        # KAON 265 (see quote above ps)
        pix_per_ap = 16.7*16.7 # previously 4 (assumed to be quadcell)
    elif wfs == 'LGSWFS':
        band = "R" # not actually at V-band
        # wavelength = 0.589e-6
        
        # side length of square subaperture (m)
        side = 0.563 
        
        # KAON 479 has CCD-39 3.0 arcsec square pixels 
        ps = 3.0
        # e-/pixel readout noise (Marcos van Dam and Bruce McIntosh - Performance of Keck AO system)
        sigma_e = 3.6 # try KAON 387 value since measurements seemed closer (frame rate)
        
        # from KAON 1303 Table 20 (1.5'' hard-coded for LGS spot size, no
        # need for Gaussian convolution)
        theta_beta = 1.93 * ( math.pi/180.0 ) / ( 60.0*60.0 ) # KAON 1317 gives 1.933 FWHM spot size from adding in quadrature the seeing disk, lenslet spot size and diffraction spot
        
        # KAON 1303 Table 7 states 0.36, but Np=1000 is already
        # measured on the detector. Modified to account for QE=0.88 
        # on the WFS detector at R-band from error budget spreadsheet
        throughput = 0.36 # * 0.88
        
        pix_per_ap = 4
    elif wfs == 'LGSWFS-OCAM2K':
        band = "R"
        # wavelength = 0.589e-6
        
        # side length of square subaperture (m)
        side = 0.563 # ~D/20 subapertures = 10.949/20 = 0.54745
        
        # from Carlos' config file
        ps = 1.5 # Noah Stiegler, 5/5/26: updated to 1.5''/px from  3.0
        sigma_e = 0.3 # e-/pixel read noise validated by telemetry ~0.3 e-/pix fwhm in the background
        
        # from KAON 1303 Table 20 -> FWHM in LGS 51 +/- 8 mas, LGS Magnitude 10.2 +/- 0.7, LGS FWHM 1.9 +/- 0.4 arcsec, LGS Small FWHM 1.5 +/- 0.3 arcsec
        theta_beta = 1.5 * ( math.pi/180.0 ) / ( 60.0*60.0 ) # 1.5 arcsec -> radians
        
        # KAON 1303 Table 8 states 0.36, but Np=1000 is already
        # measured on the detector. Modified to account for QE=0.88 
        # on the WFS detector at R-band from error budget spreadsheet
        throughput = 0.36 * 0.88 # QE ratio from old to new LGS WFS is 0.88
        
        # quadcell
        pix_per_ap = 4 * 4 # <- 4x4
    elif wfs == 'LGS-HODM-HOWFS':
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "R"
        # wavelength = 0.589e-6
        side = 0.17
        ps = 3.0
        sigma_e = 0.1
        theta_beta = 1.5 * ( math.pi/180.0 ) / ( 60.0*60.0 )
        throughput = 0.36 * 0.88
        pix_per_ap = 4 * 4 # <- 4x4
    elif wfs == 'TRICK-H':
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "H"
        # wavelength = 1.63e-6

        # side length of square subaperture (m)
        # turn into square aperture of same area as primary
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) ) # 9.571275453301643

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
        throughput *= 0.96**4 * 0.95 # 0.45 total
        
        # ROI reduces from 16x16 to 2x2 as residual is reduced
        pix_per_ap = 4 * 4 # <- 8x8
    elif wfs == 'TRICK-K':
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "K"
        # wavelength = 2.19e-6

        # side length of square subaperture (m) 
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) ) # 9.57
        
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
        throughput *= 0.96**4 * 0.95 # 0.50 total

        # ROI decreases from 16x16 to 2x2 as residual reduces
        pix_per_ap = 4 * 4 # <- 8x8 or 4x4
    elif wfs == 'TRICK-H_5rne':
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "H"
        # wavelength = 1.63e-6

        # side length of square subaperture (m)
        # turn into square aperture of same area as primary
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) ) # 9.571275453301643

        # From Carlos' config file
        ps = 0.06
        # Changed to RNE of 5 for sims of TRICK vs. TREAT on 6/8/26
        sigma_e = 5

        # Using OSIRIS FWHM from KAON 1303 Table 13 (as suggested by Peter)
        theta_beta = 0.055 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 8
        throughput = 0.56
        # Modify to add 4 lenses and a filter inside TRICK
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95 # 0.45 total
        
        # ROI reduces from 16x16 to 2x2 as residual is reduced
        pix_per_ap = 4 * 4 # <- 8x8
    elif wfs == 'TRICK-K_5rne':
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "K"
        # wavelength = 2.19e-6

        # side length of square subaperture (m) 
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) ) # 9.57
        
        # From Carlos' config file
        ps = 0.04
        # Changed to RNE of 5 for sims of TRICK vs. TREAT on 6/8/26
        sigma_e = 5
        
        # Scaling the K band 0.055 by 2.19/1.63 (wavelength ratio)        
        theta_beta = 0.074 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 8
        throughput = 0.62
        # Modify to add 4 lenses and a filter inside TRICK
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95 # 0.50 total

        # ROI decreases from 16x16 to 2x2 as residual reduces
        pix_per_ap = 4 * 4 # <- 8x8 or 4x4
    elif wfs == "TREAT-K":
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "K"
        # wavelength = 2.19e-6

        # side length of square subaperture (m) 
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) ) # 9.57
        
        # From Carlos' config file
        ps = 0.04 * (24 / 18)
        # Modified to get SNR=5 at H=15
        sigma_e = 0.5
        
        # Scaling the K band 0.055 by 2.19/1.63 (wavelength ratio)        
        theta_beta = 0.074 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 8
        throughput = 0.62
        # Modify to add 4 lenses and a filter inside TRICK
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95 # 0.50 total

        # ROI decreases from 16x16 to 2x2 as residual reduces
        pix_per_ap = 4 * 4 # <- 8x8 or 4x4
    elif wfs == "TREAT-H":
        # NEED TO ADJUST SPOT SIZE CALCULATION WHEN THIS IS USED
        band = "H"
        # wavelength = 1.63e-6

        # side length of square subaperture (m)
        # turn into square aperture of same area as primary
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) ) # 9.571275453301643

        # From Carlos' config file
        ps = 0.06 * (24 / 18)   
        # Modified to get SNR=5 at H=15
        sigma_e = .5

        # Using OSIRIS FWHM from KAON 1303 Table 13 (as suggested by Peter)
        theta_beta = 0.055 * ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 8
        throughput = 0.56
        # Modify to add 4 lenses and a filter inside TRICK
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95 # 0.45 total
        
        # ROI reduces from 16x16 to 2x2 as residual is reduced
        pix_per_ap = 4 * 4 # <- 8x8
    
    elif wfs == 'STRAP':
        band = "R"
        band_wvl = 0.641e-6
        # wavelength = 0.641e-6

        # side length of square subaperture (m)          
        side = math.sqrt( math.pi * ( (D  / 2.0)**2 - (Ds / 2.0)**2 ) )
        
        # 2014 Wizinowich paper
        # has pixel size = 1.4 '' for STRAP in Table 1
        ps = 1.4 # changed from 1.3, which I couldn't verify from original comment (from KAON 1322, just above equation 19)
        sigma_e = 0.0

        # There appears to be inconsistencies in that KAON 1322 Section 7.6
        # which quotes 3000 photons/aperture/frame (not sure what 
        # brightness star this would be for). Maybe GC R=15?
        
        # B.DiGia 12/13/2024 - changed spot size calculation
        # to account for intrinsic theta (0.625) convolved
        # with seeing-limited disk theta_r0
        theta_r0 = seeing_limit_spot_size(band_wvl, r0)
        theta_beta = np.sqrt(theta_r0**2.0 + 0.625**2.0)
        # Convert spot size to radians
        theta_beta *= ( math.pi/180.0 ) / ( 60.0*60.0 )

        # from KAON 1303 Table 7
        # Modified to account for QE=0.50 on the WFS detector at R-band
        # from error budget spreadsheet
        throughput = 0.32 # * 0.50

        # ROI
        pix_per_ap = 2 * 2 # <- 2x2

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

def keck_nea_photons_any_config(wfs:str, side:float, throughput:float, ps:float, 
                                theta_beta:float, band:str, sigma_e:float, 
                                pix_per_ap:int, time:float, m:float):
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
        Total number of pixels per sub-aperture (so a 2x2 quad cell would be 4 pixels)

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
        Number of photons from the star per pixel within subaperture

    Nb          : float
        Number of background photons per pixel within subaperture
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
    print(f"  N_photons per pixel from background (powfs.bkgrnd): {Nb:.3f}")
    print(f"  SNR: {SNR:.3f}")
    print(f"  NEA (powfs.nearecon): {sigma_theta:.3f} mas")

    return SNR, sigma_theta, Np, Nb
    
def n_photons(side:float, time:float, m:float, band:str, ps:float, 
              throughput:float):
    """
    Calculate the number of photons from a star and
    background incident on a square area in a given time 
    interval.

    By Paolo Turri
        
    Bibliography:
    [1] Bessel et al. (1998): https://articles.adsabs.harvard.edu/pdf/1998A%26A...333..231B (see Table A2)
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
             'delta_lambd': [0.0665, 0.1037, 0.0909, 0.1479, 0.1042, 0.3268, 
                             0.2607, 0.5569],
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
    f = c / (lambd * 1e-6) # lambd converted from microns to meters, c given in meters s^-1
    # Numeric flux (s^-1 cm^-2 A^-1)
    phi_n = ( phi_erg * 1e-11 ) / ( h * f )
    # Zeropoint (m = 0) number of photons on detector
    n_ph_0 = phi_n * ( (side * 1e2) ** 2) * time * delta_lamb * 1e4 * throughput  # 1e4 is microns to angstrom conversion for delta_lambd, 1e2 * side converts side in m to cm
    # Number of star photons
    n_ph_star = n_ph_0 * ( 10**(-0.4 * m) ) 

    # Number of background photons (px^-1)
    n_ph_bkg = n_ph_0 * ( 10**(-0.4 * bkg_m) ) * (ps**2.0)  
    
    return n_ph_star, n_ph_bkg

def keck_ttmag_to_itime(ttmag:float, wfs:str='strap'):
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

def gain_from_telem(snr_adu:float, snr_e:float):
    """
    Function to calculate the gain based on SNR from keck_nea_photons
    (in absence of actual gain data) and SNR from real telemetry. 

    Inputs:
    -------
    snr_adu : float
        Signal-to-noise ratio in adu from telemetry

    snr_e   : float
        Signal-to-noise ratio in electrons calculated from keck_nea_photons()

    Outputs:
    --------
    g       : float
        Gain

    By Brooke DiGia
    """
    g = ( snr_e / snr_adu )**2.0
    return g

def print_wfe_metrics(directory:str='./', seed:int=10):
    """
    Function to print various wave-front error (WFE) metrics 
    to terminal.

    Inputs:
    ------------
    directory      : string, default is current working directory
        Path to directory where simulation results live

    seed           : int, default=10
        Seed with which simulation was run

    Outputs:
    ------------
    open_mean_nm   : array, len=3, dtype=float
        Array containing WFE metrics for open-loop MAOS results
        averaged over all the PSF evalution locations.

    closed_mean_nm : array, len=3, dtype=float
        Array containing WFE metrics for closed-loop MAOS results
        averaged over all the PSF evalution locations.

    open_xx_mean_nm   : array, shape=[N,3], dtype=float
        Array containing WFE metrics for open-loop MAOS results
        evaluated at each PSF location. Shape is [N, 3] where
        N is the number of PSF locations. Will return None if
        only a single PSF location. 

    closed_xx_mean_nm : array, shape=[N,3], dtype=float
        Array containing WFE metrics for closed-loop MAOS results
        evaluated at each PSF location. Shape is [N, 3] where
        N is the number of PSF locations. Will return None if
        only a single PSF location.
    
    """
    # Field averaged results
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    print("Looking in directory:", directory)

    # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    open_mean_nm = np.sqrt(results[0].mean(axis=0)) * 1.0e9

    # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
    clos_mean_nm = np.sqrt(results[2].mean(axis=0)) * 1.0e9

    # Field-dependent resutls
    # Determine if we have a field-dependent WFE results file in extra/
    results_xx_file = f'{directory}/extra/Resp_{seed}.bin'
    if os.path.exists(results_xx_file):
        results_xx = readbin.readbin(results_xx_file)

        open_xx_mean_nm = np.zeros((results_xx[2].shape[0], 3), dtype=float)
        clos_xx_mean_nm = np.zeros((results_xx[3].shape[0], 3), dtype=float)

        # Loop through PSF positions and get RMS WFE in nm
        for xx in range(open_xx_mean_nm.shape[0]):
            
            # Open-loop WFE (nm): Piston removed, TT only, Piston+TT removed
            open_xx_mean_nm[xx] = np.sqrt(results_xx[2][xx].mean(axis=0)) * 1.0e9
            
            # Closed-loop WFE (nm): Piston removed, TT only, Piston+TT removed
            clos_xx_mean_nm[xx] = np.sqrt(results_xx[3][xx].mean(axis=0)) * 1.0e9

    else:
        results_xx = None
        open_xx_mean_nm = None
        clos_xx_mean_nm = None

    print('---------------------')
    print('WaveFront Error (nm): [note, piston removed from all]')
    print('---------------------')
    print(f'{"Field Avg":<9s}  {"Total":>11s}  {"High_Order":>11s}  {"TT":>11s}')
    print(f'{"---------":<9s}  {"-----------":>11s}  {"----------":>11s}  {"----------":>11s}')
    print(f'{"Open     ":<9s}  {open_mean_nm[0]:11.1f}  {open_mean_nm[2]:11.1f}  {open_mean_nm[1]:11.1f}')
    print(f'{"Closed   ":<9s}  {clos_mean_nm[0]:11.1f}  {clos_mean_nm[2]:11.1f}  {clos_mean_nm[1]:11.1f}')

    if results_xx != None:
        # Loop through PSF positions and print WFE metrics
        for xx in range(open_xx_mean_nm.shape[0]):
            print()
            print(f'{"Pos ":<3s} {xx:<2d}')
            print(f'{"-------":<9s}')
            print(f'{"Open   ":<9s}  {open_xx_mean_nm[xx,0]:11.1f}  {open_xx_mean_nm[xx,2]:11.1f}  {open_xx_mean_nm[xx,1]:11.1f}')
            print(f'{"Closed ":<9s}  {clos_xx_mean_nm[xx,0]:11.1f}  {clos_xx_mean_nm[xx,2]:11.1f}  {clos_xx_mean_nm[xx,1]:11.1f}')

    return open_mean_nm, clos_mean_nm, open_xx_mean_nm, clos_xx_mean_nm
    
def print_psf_metrics_x0y0(directory:str='./', oversamp:int=3, 
                           seed:int=10):
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
        mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], oversamp=oversamp)
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

def print_psf_metrics_open(directory:str='./', oversamp:int=3, 
                           seed:int=10):
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
        mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], oversamp=oversamp)
        sout  = f'{hdr["WVL"]*1e6:10.3f} '
        sout += f'{mets["strehl"]:6.2f} '
        sout += f'{mets["emp_fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["fwhm"]*1e3:10.1f} ' 
        sout += f'{mets["ee80"]*1e3:6.1f}' 
        print(sout)

    psf_all_wvls.close()
    return

def get_psf_metrics_over_field(directory:str='./', oversamp:int=3, 
                               seed:int=10, cut_radius:int=30):
    """
    Print some PSF metrics vs. wavelength and field position for PSFs
    computed by MAOS. Closed-loop.

    Inputs:
    ------------
    directory        : string, default is current directory
        Directory where MAOS simulation results live

    oversamp         : int, default=3

    cut_radius       : int, default=30 pixels

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
    fits_files = glob.glob(directory + f'evlpsfcl_{seed}_x*_y*.fits')
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)
    npos = len(fits_files)

    psf_all_wvls.close()

    xpos = np.zeros((npos, nwvl), dtype=float)
    ypos = np.zeros((npos, nwvl), dtype=float)
    wavelengths = np.zeros((npos, nwvl), dtype=float)
    strehl_values = np.zeros((npos, nwvl), dtype=float)
    fwhm_gaus_values = np.zeros((npos, nwvl), dtype=float)
    fwhm_emp_values = np.zeros((npos, nwvl), dtype=float)
    r_ee50_values = np.zeros((npos, nwvl), dtype=float)
    r_ee80_values = np.zeros((npos, nwvl), dtype=float)
 
    for xx in range(npos):
        psf_all_wvls = fits.open(fits_files[xx])

        file_name = fits_files[xx].split('/')[-1]
        # file_root = file_name.split('.')[0]
        file_root = str(Path(file_name).stem)
        tmp = file_root.split('_')
        tmpx = float(tmp[2][1:])
        tmpy = float(tmp[3][1:])
        print('xx = ', tmpx, 'yy = ', tmpy)

        for pp in range(nwvl):
            xpos[xx, pp] = tmpx
            ypos[xx, pp] = tmpy
            
            psf = psf_all_wvls[pp].data
            hdr = psf_all_wvls[pp].header
            mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], 
                                                   oversamp=oversamp,
                                                   cut_radius=cut_radius)
            wavelengths[xx, pp] = hdr["WVL"] * 1.0e6
            strehl_values[xx, pp] = mets["strehl"]
            fwhm_gaus_values[xx, pp] = mets["emp_fwhm"] * 1.0e3
            fwhm_emp_values[xx, pp] = mets["fwhm"] * 1.0e3
            r_ee50_values[xx, pp] = mets["ee50"] * 1.0e3
            r_ee80_values[xx, pp] = mets["ee80"] * 1.0e3


        psf_all_wvls.close()
        
    return xpos, ypos, wavelengths, strehl_values, fwhm_gaus_values, fwhm_emp_values, r_ee50_values, r_ee80_values

def read_maos_psd(psd_input_file:str, type:str='jitter'):
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
    
def psd_add_vibrations(psd_input_file:str, vib_freq:float, 
                       vib_jitter_amp:float):
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
    
def psd_integrate_sqrt(freq:float, psd:list):
    """
    Function to integrate and take the square root of a PSD to give the total 
    WFE or jitter.

    Inputs:
    ------------
    freq      : float
        Frequency (Hz) array

    psd       : 1D array, dtype=float
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

def calc_strehl(sim_dir:str, out_file:str, skysub:bool=False, 
                sim_seed=1, apersize=0.6, verbose:bool=False):
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

    aper_size        : float, default = 0.6 arcsec ** after spot check
        The aperture size over which to calculate the Strehl and FWHM

    verbose          : boolean, default = False
        Option to turn on verbose output

    Outputs:
    ------------
    strehl_to_return : array, len(nwvl), dtype=float
        Strehl values at each wavelength for which MAOS was run

    fwhm_to_return   : array, len(nwvl), dtype=float
        FWHM values at each wavelength

    rmswfe_to_return : array, len(nwvl), dtype=float
        RMS WFE values at each wavelength

    emp_fwhm_to_return : array, len(nvwl), dtype-float
        Empirical FWHM (mas)

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
    path = f'{sim_dir}evlpsfcl_{sim_seed}_x0_y0.fits'
    print(path)
    fits_files = glob.glob(path)
    psf_all_wvls = fits.open(fits_files[0])
    nwvl = len(psf_all_wvls)

    # Get the diffraction limited image from the input simulation directory
    dl_img_files = glob.glob(sim_dir + 'evlpsfdl.fits')
    dl_all_wvls = fits.open(dl_img_files[0])

    if verbose:
        print("Lambda (micron) | Strehl | RMS WFE (nm) | FWHM (mas)")
        print("----------------------------------------------------")

    strehl_to_return = np.zeros(nwvl)
    fwhm_to_return = np.zeros(nwvl)
    rmswfe_to_return = np.zeros(nwvl)
    emp_fwhm_to_return = np.zeros(nwvl)
    # Loop over the MAOS PSF stack to calculate metrics
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
        strehl, fwhm, rmswfe, emp_fwhm = calc_strehl_single(sim_dir, psf, hdr, 
                                                            radius, skysub, 
                                                            dl_peak_flux_ratio)

        # mets = metrics.calc_psf_metrics_single(psf, hdr['DP'], oversamp=1) # default oversamp value is 3

        strehl_to_return[i] = strehl
        fwhm_to_return[i] = fwhm
        rmswfe_to_return[i] = rmswfe
        emp_fwhm_to_return[i] = emp_fwhm

        _out.write(fmt_dat.format(img=psf_lambda, strehl=strehl, 
                                  rms=rmswfe, fwhm=fwhm))
        
        if verbose:
            print(fmt_dat.format(img=psf_lambda, strehl=strehl, 
                                 rms=rmswfe, fwhm=fwhm), 
                                 end="")
            print(">> PSF peak flux value: %0.6f\n" % psf.max())

    # Close output file
    _out.close()
    return strehl_to_return, fwhm_to_return, rmswfe_to_return, emp_fwhm_to_return

def calc_strehl_single(sim_dir:str, psf:list, hdr:dict, 
                       radius:float, skysub:bool, 
                       dl_peak_flux_ratio:float):
    """
    Modified from KAI by Brooke DiGia 
    (https://github.com/Keck-DataReductionPipelines/KAI/tree/dev)
    for use on MAOS-generated PSFs and diffraction-limited images.

    Function to calculate the Strehl for a single image PSF.

    Inputs:
    ----------
    sim_dir  : str
        Simulation directory, output will be stored here

    psf      : array
        PSF data

    hdr      : dictionary 
        FITS header associated with PSF data

    radius   : float
        Extraction radius in fitting

    skysub   : boolean 
        True to perform sky subraction on PSF

    Outputs:
    ------------
    strehl   : float
        Calculated Strehl value

    fwhm     : float
        Full-width at half-maximum of Gaussian fitted to input PSF

    rms_wfe  : float
        Root-mean-square wave-front error (nm)

    emp_fwhm : float
        Empirical FWHM calculated directly from pixels in mas
    """
    wavelength = hdr["WVL"]*1.0e6 # microns
    scale = hdr["DP"]             # arcsec/px

    # Coordinates of Strehl source (MAOS PSFs are output such that the
    # Strehl source is always centered in the image)
    coords = np.array([psf.shape[0]/2.0 , psf.shape[1]/2.0])

    # First estimate the DL FWHM in pixels. Use this to set the initial boxsize 
    # for the FWHM estimation...note that this is NOT the aperture size 
    # specified above, which is only used for estimating the Strehl:
    
    # Keck telescope diameter in meters
    telescope_diam = 10.5
    dl_res_in_pix = ( 0.25 * wavelength ) / ( telescope_diam * scale )
    print(f"Diffraction-limited resolution [px] = {dl_res_in_pix} | Scale = {scale} arcsec/px")
    fwhm_min = dl_res_in_pix # 0.9
    fwhm_max = 100.0
    fwhm = 0.0
    emp_fwhm = 0.0
    fwhm_boxsize = int( np.ceil( ( 4 * dl_res_in_pix ) ) )
    if fwhm_boxsize < 3:
        fwhm_boxsize = 3
    pos_delta_max = 2 * fwhm_min
    box_scale = 1.0
    iters = 0
    
    # Steadily increase the boxsize until we get a reasonable FWHM
    while ( (fwhm < fwhm_min) or (fwhm > fwhm_max) ) and (iters < 50): # bumped iters up from 30 to 50
        box_scale += iters * 0.1
        iters += 1
        g2d = fit_gaussian2d(psf, coords, fwhm_boxsize * box_scale, 
                             fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                             pos_delta_max=pos_delta_max)
        sigma = (g2d.x_stddev_0.value + g2d.y_stddev_0.value) / 2.0
        fwhm = stddev_to_fwhm(sigma)
        emp_fwhm = empirical_fwhm(psf, scale)
        print(f"FWHM on iteration {iters} = {fwhm*scale*1.0e3:.2f} mas | Empirical FWHM on iteration {iters} = {emp_fwhm*1.0e3:.2f} mas")

        # Update the coordinates if they are reasonable. 
        if ((np.abs(g2d.x_mean_0.value - coords[0]) < fwhm_boxsize) and
            (np.abs(g2d.y_mean_0.value - coords[1]) < fwhm_boxsize)):
            coords = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])
            print(np.array([g2d.x_mean_0.value, g2d.y_mean_0.value]))

    # Convert to milli-arcseconds
    fwhm *= scale * 1e3
    emp_fwhm *= 1.0e3

    # metrics = fit_gaussian2d_alternative(psf, coords, scale)
    # fwhm = metrics['fwhm']*1e3 # mas
    # emp_fwhm = metrics['emp_fwhm']*1e3 # mas

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
        emp_fwhm = -1.0

    return strehl, fwhm, rms_wfe, emp_fwhm

def calc_peak_flux_ratio(sim_dir:str, img:list, coords:list, 
                         radius:int, wavelength:float, 
                         skysub:bool):
    """
    Modified from KAI by Brooke DiGia for use on MAOS-generated PSFs.
    Function to calculate the ratio of peak flux in the input PSF image
    to the sum of the PSF pixel values. Optional plotting routine to
    visually inspect the sky subtraction annulus (if skysub is turned on).

    Inputs:
    ------------
    sim_dir         : str
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

def fit_gaussian2d(img:list, coords:list, boxsize:int, plot:bool=False, 
                   fwhm_min:float=1.7, fwhm_max:float=30, 
                   pos_delta_max:float=1.7):
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

def fit_gaussian2d_alternative(psf:list, coords:list, 
                               pixel_scale:float, 
                               cut_radius:int=20, 
                               oversamp:int=3, 
                               plot:bool=False):
    """
    Calculate the FWHM of an objected located at the pixel
    coordinates in the image. The FWHM will be estimated 
    from a cutout with the specified boxsize. Adopted from
    the PAARTI metrics module (see metrics.calc_psf_metrics_single()).

    Inputs:
    ------------

    coords        : len=2 ndarray
        The [x, y] pixel position of the star in the image

    Outputs:
    ------------
    g2d           : Gaussian model object
        2D Gaussian fit
    """
    # Cutout and oversample the image. 
    # Odd box, with center in middle pixel.    
    psf_c = psf[int(coords[1]-cut_radius) : int(coords[1]+cut_radius+1),
                int(coords[0]-cut_radius) : int(coords[0]+cut_radius+1)]
    if oversamp > 1:
        psf_co = scipy.ndimage.zoom(psf_c, oversamp, order=1)
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
        enc_energy[rr] = energy

    # Normalize the encircled energy by the total. Not quite correct,
    # but close enough.
    tot_energy = psf_co.sum() * oversamp**2
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
    if (ee25_rad == np.nan) and (ee50_rad == np.nan) and (ee80_rad == np.nan):
        return
    else:
        print("Valid energy radii values. Proceeding with fit...")
        g2d_model = models.Gaussian2D(1.0, psf_co.shape[0]/2.0, psf_co.shape[1]/2.0,
                                      ee25_rad, ee25_rad, theta=0,
                                      bounds={'x_stddev':[0.1, ee80_rad],
                                              'y_stddev':[0.1, ee80_rad],
                                              'amplitude':[0.001, 2]})
        c2d_model = models.Const2D(amplitude=0.0)
            
        model = g2d_model + c2d_model
        fitter = fitting.LevMarLSQFitter()
    
        y2d, x2d = np.mgrid[:psf_co.shape[0], :psf_co.shape[1]]
        print(y2d, np.where(np.isnan(y2d) == True))
        print(x2d, np.where(np.isnan(x2d) == True))
        print(psf_co, np.where(np.isnan(psf_co) == True))
        g2d_params = fitter(model, x2d, y2d, psf_co)
    
        # Save the FWHM and angle. In oversamp pixels.
        x_fwhm = stddev_to_fwhm(g2d_params.x_stddev_0.value) # * stats.gaussian_sigma_to_fwhm
        y_fwhm = stddev_to_fwhm(g2d_params.y_stddev_0.value) # * stats.gaussian_sigma_to_fwhm
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
    
        if plot:
            mod_img = g2d_params(x2d, y2d)
            plt.figure(1, figsize=(15,5))
            plt.clf()
            plt.subplots_adjust(left=0.05, wspace=0.3)
            plt.subplot(1, 3, 1)
            plt.imshow(mod_img, vmin=mod_img.min(), vmax=mod_img.max(),
                       origin='lower')
            plt.colorbar()
            plt.title("Original")
            
            plt.subplot(1, 3, 2)
            plt.imshow(psf_co, vmin=mod_img.min(), vmax=mod_img.max(),
                       origin='lower')
            plt.colorbar()
            plt.title("Model")
            
            plt.subplot(1, 3, 3)
            plt.imshow(psf_co - mod_img, origin='lower')
            plt.colorbar()
            plt.title("Orig - Mod")
            plt.show()
        
        return results

def empirical_fwhm(psf:list, pixel_scale:float):
    """
    Function to calculate the FWHM of an image (i.e. PSF) from the image pixel values, 
    as opposed to a model fit (e.g. Gaussian fit)

    Inputs:
    -------
    psf         : 2D array, dtype=float
        Image data array

    pixel_scale : float
        Pixel scale (arcsec/px) from FITS image header (on sky or MAOS)
    
    Outputs:
    --------
    emp_fwhm : float
        Empirical FWHM in arcsec

    """
    # Max value (brightest pixel flux) and index
    max_flux = np.max(psf)
    half_max = max_flux / 2.0
    hmi = np.where(psf >= half_max)
    area_count = len(hmi[0])
    emp_fwhm = 2.0 * (area_count / np.pi)**0.5

    # Return FWHM in arcsec (use pixel scale to convert from px to arcsec)
    return emp_fwhm * pixel_scale 
    
def fwhm_to_stddev(fwhm:float):
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

def stddev_to_fwhm(stddev:float):
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

@u.quantity_input(w_mass=u.m**(1/3))
def fried(DIMM:u.arcsec, w_mass:list, airmass:float, wvl:u.nm=500.0*u.nm) -> u.m:
    """
    Function to calculate the Fried parameter r0z given the total seeing
    in arcseconds.

    Inputs:
    ------------
    DIMM    : float
        DIMM seeing, arcsec 
    
    w_mass  : 1D array, len(6), dtype = float
        Array of MASS weights, [m^(1/3)]

    airmass : float
        Airmass (sec(observation angle))

    wvl     : float, default = 500 nm
        Wavelength in nm

    Outputs:
    ------------
    r0z     : float
        The Fried parameter, r0z, in meters. Currently returning the KAON version.

    By Brooke DiGia
    """ 
    # KAON r0 equation (Roddier 1981's equation, using DIMM measurement as full atm seeing)
    r0z1 = 0.976 * ( wvl.to(u.m) / arcsec_to_rad(DIMM).to('', equivalencies=u.dimensionless_angles() ))

    # Claire Max and Roddier 1981 definition 
    # (slightly different prefactors of 0.423 and 2.905/6.88 ~= 0.422 respectively)
    k = ( 2.0 * np.pi ) / wvl.to(u.m)
    r0z2 = ( (2.905/6.88) * k**2 * float(airmass) * np.sum(w_mass) )**(-3.0/5.0)
    return r0z1

@u.quantity_input(mass=u.m**(1/3), windspd=u.m/u.s)
def tau0(dimm:u.arcsec, mass:list, windspd:list, airmass:float, wvl:u.nm=500.0*u.nm) -> u.s:
    """
    Function to calculate the atmospheric coherence time tau0

    Inputs:
    --------
    dimm        : float
        Total seeing (arcsec) from DIMM data file
    
    mass        : 7-entry 1D array, floats
        c_l weights from MASS data file. Response from 
        Mark Chun verifies that MASS file entries are indeed Cn^2*delta_h,
        as opposed to pure Cn^2 values. Cn^2*delta_h = c_l. NOTE: the
        last element of this array is the total MASS seeing (not a weight),
        so it is not used in the calculation

        hts = [100.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0] m

    windspd     : array, dtype=float
        Wind speed profile at heights h 

    airmass     : float
        Airmass, required for r0 calculation within estimate_turbulence()

    wvl         : float, default = 500
        Wavelength in nm for which to calculate coherence time. Convention is
        500 nm

    Outputs:
    --------
    tau         : float
        Atmospheric coherence time tau0 in seconds 
        (named tau to avoid overloading function name) 
        as defined by Travouillon et al 2009. See also equation 1 at:
        https://arxiv.org/pdf/1101.3211

    By Brooke DiGia
    """
    # Known heights above telescope (m)
    hts = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
    r0, cl = estimate_turbulence(dimm, mass, airmass, wvl=wvl, normalize=False)
    approx = np.sum(np.multiply(cl, np.abs(windspd)**(5.0/3.0)))

    # Convert nm to m
    # tau = 0.057 * wvl**(6.0/5.0) * approx**(-3.0/5.0)

    # Calculate tau using second formula 
    # (Roddier 1981, http://www.eso.org/gen-fac/pubs/astclim/papers/venice2001/venice2001-msarazin.pdf, 
    # equations 3-4), also in Claire Max's notes equation 5: 
    # https://www.ucolick.org/~max/289/Assigned%20Readings/Max_Adaptive_Optics_Intro_v1.pdf
    Vbar = ( approx / np.sum(cl) )**(3.0/5.0)
    tau2 = 0.31 * ( r0  / Vbar )
    return tau2

@u.quantity_input(mass=u.m**(1/3))
def theta0(dimm:u.arcsec, mass:list, airmass:float, wvl:u.nm=500.0*u.nm) -> u.arcsec:
    """
    Function to calculate the isoplanatic angle theta0 
    (http://www.ctio.noirlab.edu/~atokovin/tutorial/part1/turb.html, equation 12)

    See equation 7 of Claire Max's AO notes: 
    https://www.ucolick.org/~max/289/Assigned%20Readings/Max_Adaptive_Optics_Intro_v1.pdf

    Inputs:
    --------
    dimm        : float
        Total seeing (arcsec) from DIMM data file
    
    mass        : 7-entry 1D array, floats
        c_l weights from MASS data file. Response from 
        Mark Chun verifies that MASS file entries are indeed Cn^2*delta_h,
        as opposed to pure Cn^2 values. Cn^2*delta_h = c_l. NOTE: the
        last element of this array is the total MASS seeing (not a weight),
        so it is not used in the calculation

        hts = [100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0] m

    airmass     : float
        Airmass, required for r0 calculation within estimate_turbulence() 

    wvl         : float, default = 500
        Wavelength at which to calculate angle (nm). Convention is 500 nm

    Outputs:
    --------
    theta0      : float
        Isoplanatic angle in arcsec

    By Brooke DiGia
    """
    # Known heights above telescope (m)
    hts = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]) * u.m
    # Calculate Fried param r0 (m) and un-normalized 7-layer turbulence profile
    # (ground layer + MASS, un-normalized for purpose of calculations of theta0
    # below)
    r0, cl = estimate_turbulence(dimm, mass, airmass, wvl=wvl, normalize=False) 

    numerator = np.sum(np.multiply(cl, hts**(5.0/3.0)))
    denominator = np.sum(cl)
    hbar = ( numerator / denominator )**(3.0/5.0) # [m]
    # Claire Max
    theta_max = 0.314 * ( 1 / airmass ) * ( r0 / hbar )

    # Calculate theta0 using Roddier 1981 
    # (ref here in equation 7, 
    # original paper in citations of: 
    # http://www.eso.org/gen-fac/pubs/astclim/papers/venice2001/venice2001-msarazin.pdf)
    k = ( 2.0 * np.pi ) / wvl.to(u.m)
    theta_roddier = ( 2.05 * k**2.0 * airmass**(8.0/3.0) * numerator )**(-3.0/5.0)
    # print(f"Theta0 Roddier = {theta_roddier.to(u.arcsec, equivalencies=u.dimensionless_angles())} | Theta0 Claire Max = {theta_max.to(u.arcsec, equivalencies=u.dimensionless_angles())}")
    return theta_roddier.to(u.arcsec, equivalencies=u.dimensionless_angles())
                            
@u.quantity_input
def arcsec_to_rad(x:u.arcsec) -> u.rad:
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
    return x

@u.quantity_input(mass_wts=u.m**(1/3))
def estimate_turbulence(dimm:u.arcsec, mass_wts:list, airmass:float, date:str=None, 
                        plot:bool=False, wvl:u.nm=500.0*u.nm, normalize:bool=True):
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
        last element of this array is the total MASS seeing (not Cn2dh),
        so it is not used in the calculation

    date      : string
        Date of MASS/DIMM data; used only for plot filename and labels

    plot      : boolean
        Option to plot turbulence profile after calculation and save to
        current working directory

    airmass   : float
        Airmass (sec(observation angle))

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
    # Strip off MASS total seeing
    mass_wts = mass_wts[:-1]
    # Calculate Fried parameter and 0th order turbulence moment
    r0 = fried(dimm, mass_wts, airmass, wvl)
    mu0 = 0.06 * wvl**2.0 * r0**(-5.0/3.0)

    c0 = abs(mu0 - np.sum(mass_wts))
    cl = np.zeros(len(mass_wts) + 1)
    if normalize:
        to_normalize = np.insert(mass_wts, [0], c0)
        tot = np.sum(to_normalize)

        # Re-normalize turbulence weights with new ground layer entry
        cl = to_normalize/tot
    else:
        cl = np.insert(mass_wts, [0], c0)

    if plot:
        hts = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 
                        16000.0])
        plt.plot(hts, cl, "ro")
        plt.title("Turbulence profile %s" % date)
        plt.ylabel(r"c_{l} coefficients")
        plt.xlabel("Height above telescope (meters)")
        plt.show()
        plt.savefig("turb_profile_%s" % date)

    return r0, cl

def fetch_mass_dimm_cfht_loc(dimm_in_hrs:float, mass_in_hrs:float, 
                             cfht_in_hrs:float, timestamp:float):
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

def remove_keywords(file:str, *args:str, verbose:bool=False):
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

    *args      : string(s)
        Keyword(s) to be removed from input FITS file header

    verbose    : boolean, default = False
        Option to turn on verbose output

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
            if verbose:
                print("Removing keyword '%s' from %s..." % (word, file))
            del hdu.header[word]

        # Overwrite original input file with desired changes
        hdu.writeto(file, mode='update')
    return

def estimate_on_sky_conditions(file:str, saveto:str, verbose:bool=False, plot:bool=False):
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
    file                   : str
        Path to on-sky FITS file

    saveto                 : str
        Path of location to save MASS/DIMM data files

    verbose                : boolean, default = False
        Option to turn on verbose terminal output

    plot                   : boolean, default = False
        Option to plot turbulence profile and save to current working 
        directory

    Outputs:
    ------------
    r0_start               : float
        Fried parameter closest to exposure start in meters

    start_turb             : array, len=7, dtype=float
        Full turbulence profile (ground layer + free atmosphere)

    wind_spd_profile       : array, len=7, dtype=float
        Full wind speed profile

    wind_dir_profile       : array, len=7, dtype=float
        Full wind direction profile

    closest_dimm_start     : float
        DIMM ('') closest to exposure start

    mass_profile_start[-1] : float
        Total MASS seeing ('') closest to exposure start

    time_of_dimm           : string
        Time when extracted DIMM was measured in HH:MM:SS

    time_of_mass           : string
        Time when extracted MASS was measured in HH:MM:SS

    tau0                   : float
        Atmospheric coherence time

    theta0                 : float
        Isoplanatic angle

    sigma_DM               : float
        DM fitting error in nm
    
    By Brooke DiGia
    """
    if verbose:
        print("NOTE: Results for MAOS configuration files marked with ***\n")
    
    with fits.open(file) as fits_file:
        hdu = fits_file[0]
        psf = hdu.data
        hdr = hdu.header
        
        # Date of observation, parsed into year, month, day (UT)
        date = hdr["DATE-OBS"]
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
                if verbose:
                    print(f"{dimmdat} saved to {saveto}")
            except Exception as error:
                print(f"Error while downloading {dimmdat} from {url}:", 
                      type(error).__name__, error)
                return
        else:
            if verbose:
                print(f"{dimmdat} exists in directory {saveto}, not downloading.")
            pass

        # Reset url
        url = url_root + "masspro/" + masspro
        if not os.path.exists(saveto + masspro):
            try:
                # Pull and save MASS file
                urllib.request.urlretrieve(url, saveto + masspro)
                if verbose:
                    print(f"{masspro} saved to {saveto}")
            except Exception:
                print(f"Error while downloading {masspro} from {url}:", 
                      type(error).__name__, error)
                return
        else:
            if verbose:
                print(f"{masspro} exists in directory {saveto}, not downloading.")
            pass

        # Pull CFHT data based on year of observation date
        cfht_url = "http://mkwc.ifa.hawaii.edu/archive/wx/cfht/cfht-wx.%s.dat" % year
        cfht = "cfht-wx.%s.dat" % year
        if not os.path.exists(saveto + cfht):
            try:
                urllib.request.urlretrieve(cfht_url, saveto + cfht)
                if verbose:
                    print(f"{cfht} saved to {saveto}")
            except Exception as error:
                print(f"Error while downloading {cfht} from {cfht_url}:",
                      type(error).__name__, error)
                return
        else:
            if verbose:
                print(f"{cfht} exists in directory {saveto}, not downloading.")
            pass

        # Pull PHTO station data based on year of observation date
        phto_url = f"http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}00&TO={day}00&STNM=91285"

        phto = "phto.%s.dat" % date_for_massdimm
        phto_clean = "phto.%s_cleaned.dat" % date_for_massdimm
        if not os.path.exists(saveto + phto_clean):
            try:
                urllib.request.urlretrieve(phto_url, saveto + phto)
                to_save = ''
                with open(saveto + phto) as html:
                    soup = BeautifulSoup(html, 'html.parser')
                    try:
                        to_save = soup.find_all('pre')[0]
                    except Exception as error:
                        print(f"Error while parsing html soup: {error}")
                        to_save = "No information"
                        for lines in to_save:
                            with open(saveto + phto_clean, 'w') as f:
                                f.write(line)
                for lines in to_save:
                    with open(saveto + phto_clean, 'w') as f:
                        line = str(lines.text)
                        f.write(line)
                if verbose:
                    print(f"{phto} saved to {saveto}")
            except Exception as error:
                print(f"Error while downloading {phto} from {phto_url}:",
                      type(error).__name__, error)
                return
        else:
            if verbose:
                print(f"{phto} exists in directory {saveto}, not downloading.")
            pass

        # Exposure time on this date, parsed into hour, minute, second (UT)
        if hdr["CURRINST"] == 'OSIRIS':
            expstart = hdr['UTC']
            expstop = hdr['UTC']
            expstop_sec_old = float(expstop[6:8])
            expstop_sec_new = expstop_sec_old + hdr['TRUITIME']
            expstop = expstop[0:6] + f'{expstop_sec_new:.2f}'
        elif hdr["CURRINST"] == 'NIRC2': # NIRC2
            expstart = hdr["EXPSTART"]
            expstop = hdr["EXPSTOP"]
        else:
            raise NotImplementedError(f"Instrument '{hdr['CURRINST']}' is not yet supported.")
            
        expstart_hr = float(expstart[:2])
        expstart_min = float(expstart[3:5])
        expstart_sec = float(expstart[6:8])
        expstop_hr = float(expstop[:2])
        expstop_min = float(expstop[3:5])
        expstop_sec = float(expstop[6:8])

        if verbose:
            print("\nDate of observation is  %s (UT)" % date)
            print("Exposure time is\t%s to %s (UT)\n" % (expstart, expstop))
        
        
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
        try:
            phto_table = read_csv(saveto + phto_clean, delim_whitespace=True, 
                              usecols=[1,6,7], skiprows=[0,1,2,3,4], 
                              names=['hght', 'drct', 'sknt'], 
                              skipfooter=1, engine='python')
            phto_hghts = np.asarray(phto_table['hght'], dtype=float)
            phto_wddir = np.array(phto_table['drct'])
            phto_wdspd = np.array(phto_table['sknt']) * ( 1852.0 / (60.0 * 60.0) )
        except Exception as error:
            file = open(saveto + phto_clean, 'r')
            content = file.read()
            file.close()
            if content == "No information":
                print("PHTO Information Not Currently Available")
            phto_hghts = [0.0]
            phto_wddir = [0.0]
            phto_wdspd = [0.0]

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
        if verbose:
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
        if verbose:
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
        
        # Estimate turbulence for beginning and end of exposure (at default wavelength of 500 nm)
        r0_start, start_turb = estimate_turbulence(closest_dimm_start*u.arcsec, 
                                                   mass_profile_start*(u.m**(1/3)),
                                                   float(hdr['AIRMASS']),
                                                   date_for_massdimm, 
                                                   plot)
        # r0_end, end_turb = estimate_turbulence(closest_dimm_stop*u.arcsec, 
        #                                        mass_profile_stop*(u.m**(1/3)),
        #                                        float(hdr['AIRMASS']),
        #                                        date_for_massdimm, plot)

        # Average CFHT data across exposure to calculate ground layer wind speed
        # and direction
        cfht_wdspd_start = cfht_wdspd[i_cfht_start]
        cfht_wddir_start = cfht_wddir[i_cfht_start]
        cfht_wdspd_stop = cfht_wdspd[i_cfht_stop]
        cfht_wddir_stop = cfht_wddir[i_cfht_stop]
        if verbose:
            print("Closest CFHT data to beginning of exposure: %0.4f (m/s) | %0.4f (deg) at %s" % 
                  (cfht_wdspd_start, cfht_wddir_start, cfht_date_start))
            print("Closest CFHT data to end of exposure:\t    %0.4f (m/s) | %0.4f (deg) at %s\n" %
                  (cfht_wdspd_stop, cfht_wddir_stop, cfht_date_stop))

        free_atm_wdspd = phto_wdspd[phto_indices]
        free_atm_wddir = phto_wddir[phto_indices]
        # Use CFHT data at exposure start rather than averaged result over exposure, 02/29/2024 B. DiGia
        wind_spd_profile = np.concatenate( (np.array([cfht_wdspd_start]), 
                                                     free_atm_wdspd) )
        wind_dir_profile = np.concatenate( (np.array([cfht_wddir_start]), 
                                                     free_atm_wddir) )
        
        # Calculate atmospheric coherence time tau_0 (s)
        tau_0 = tau0(closest_dimm_start*u.arcsec, mass_profile_start*(u.m**(1/3)), 
                     wind_spd_profile*(u.m/u.s), hdr['AIRMASS'], wvl=500.0*u.nm)

        # Calculate isoplanatic angle theta0
        theta_0_roddier = theta0(closest_dimm_start*u.arcsec, mass_profile_start*(u.m**(1/3)), 
                                 hdr['AIRMASS'], wvl=500.0*u.nm)
        
        # Calculate expected DM fitting error (to be compared to MAOS DM fitting
        # error simulation)
        sigma_DM = DM_fitting_error(r0_start)
        
        if verbose:
            print("Free atm wind speed/direction profiles taken at these heights:", 
                  phto_hghts[phto_indices])
            print("*** Full wind speed profile (m/s): ", wind_spd_profile)
            print("*** Full wind direction profile (deg): ", wind_dir_profile)
            print(f"AIRMASS = {hdr['AIRMASS']} --> Zenith angle (deg): {np.degrees(np.arccos(1.0/float(hdr['AIRMASS'])))}")
        
    # 02/29/2024, B. DiGia - returning quantities closest to the exposure start rather than averaging over
    # exposure
    time_of_dimm = f"{dimm_hr[i_dimm_start]}:{dimm_min[i_dimm_start]}:{dimm_sec[i_dimm_start]}"
    time_of_mass = f"{mass_hr[i_mass_start]}:{mass_min[i_mass_start]}:{mass_sec[i_mass_start]}"
    return r0_start, start_turb, wind_spd_profile, wind_dir_profile, closest_dimm_start, mass_profile_start[-1], time_of_dimm, time_of_mass, tau_0, theta_0_roddier, sigma_DM

def maos_windshake_grid(amps:list, on_sky:list, thres:float=0.05):
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
    from paarti.utils import keck_utils
    base_root = "/u/bdigia/work/ao/keck/maos/keck/my_base/"
    best = []

    # Convert on_sky array to numpy array in case user
    # did not input it as such
    on_sky = np.array(on_sky)

    from paarti.utils import keck_utils

    for amp in amps:
    	for i in range(on_sky.shape[0]):
            # Get atmospheric conditions for current on_sky frame
            fried, turbpro, windspd, winddrct, _, _, _, _, _ = estimate_on_sky_conditions(on_sky[i][1]+on_sky[i][0]+"_psf.fits", 
                                                                                          on_sky[i][1])
            
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
                                                                 skysub=False, 
                                                                 apersize=0.6)

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

def maos_phase_screen_grid(r0s:list, l0s:list, on_sky:list, base_root:Path, 
                           thres:float=0.05):
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

    baseroot     : Path
       Path object for directory from which MAOS sims are run (e.g. MAOS /base/)

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
    best = []

    # Convert on_sky array to numpy array in case user
    # did not input it as such
    on_sky = np.array(on_sky)

    for r0 in r0s:
        for l0 in l0s:
            for i in range(on_sky.shape[0]):
                # Get atmospheric conditions for current on_sky frame
                fried, turbpro, windspd, winddrct, _, _, _, _, _ = estimate_on_sky_conditions(on_sky[i][1]+on_sky[i][0]+"_psf.fits", 
                                                                                              on_sky[i][1])

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
                                                                     apersize=0.6)
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

def maos_comp_to_sky_plot(metric:str, sim_dirs:list, sky_metrics:list, saveto:str, 
                          file_suffix:str=".pdf"):
    """
    Function to take an array of MAOS simulation results
    and plot the metric results at 2.12 microns versus the 
    corresponding on-sky metric results (available at only
    2.12 microns). 

    Inputs:
    ----------
    metric      : str
	Which metric to make the plot for (Strehl or FWHM)

    sim_dirs    : array, variable length, dtype=str
        Array of MAOS simulation output folder names.
        E.g. "~/base/A_keck_scao_lgs_gc". Don't forget
        the trailing slash! FULL paths

    sky_metrics : array, len(sim_dirs), dtype=float
	Array of on-sky Strehl values, to be plotted against
        MAOS Strehls

    saveto      : str
        Location to save plot images

    file_suffix : str, default=".pdf"
        File type for saving plot (PDF or PNG)

    Outputs:
    ----------
    None, plot image (PDF or PNG) saved to desired output
    folder

    By Brooke DiGia
    """
    if (metric == "Strehl") or (metric == "strehl"):
        filename = "MAOS_PAARTI_Strehl_vs_Sky"
    elif (metric == "FWHM") or (metric == "fwhm"):
        filename = "MAOS_PAARTI_FWHM_vs_Sky"
    else:
        print("Error: Invalid metric entered. Must be Strehl or FWHM")
        return 
        
    maos = np.zeros(len(sim_dirs))
    for i in range(len(sim_dirs)):
        metrics_output = sim_dirs[i][:-1] + "_sim_results.txt"
        print(f"Grabbing results from {sim_dirs[i]}...")

	# Calculate Strehl + metrics for ith MAOS simulation
        strehl_array, fwhm_array, rmswfe_array = calc_strehl(sim_dirs[i], metrics_output,
                                                             skysub=False, apersize=0.6)
        
	# Grab 2.12 micron metric only
        if (metric == "Strehl") or (metric == "strehl"):
            maos[i] = strehl_array[-1]
        elif (metric == "FWHM") or (metric == "fwhm"):
            maos[i] = fwhm_array[-1]
    
    # Plotting
    _, ax = plt.subplots()
    ax.set_xlabel(f"MAOS {metric} (calculated via PAARTI)")
    ax.set_ylabel(f"On-Sky GC {metric}")
    ax.set_title(f"On-Sky {metric} versus MAOS {metric}")
    ax.plot(maos, sky_metrics, "bo")
    ax.axline((0, 0), slope=1)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.5)
    plt.savefig(saveto + filename + file_suffix)
    return

def maos_metric_plot(metric:str, sim_dirs:list, saveto:str, nwvl:int=5, 
                     file_suffix:str=".pdf"):
    """
    Function to parse MAOS simulation results and make a metric 
    (e.g. Strehl/FWHM) vs. wavelength plot for all simulated
    wavelengths. The metric data is averaged over multiple simulations
    (e.g. for 2.12 microns, multiple MAOS sims are averaged to compute
    the Strehl/FWHM, and the standard deviation of these measurements is
    included as an error bar.

    Inputs:
    ----------
    metric      : str
	String identifier for which type of metric to make the plot.
        Valid metrics include: Strehl, FWHM, RMS WFE, and R_EE80
    
    sim_dirs    : array, variable length, dtype=string
        Array of MAOS simulation output folder names.
        E.g. "~/base/A_keck_scao_lgs_gc". Don't forget
        the trailing slash! FULL paths

    saveto      : str
        Location to save plot images

    nwvl        : int
	Number of wavelengths in each MAOS simulation

    file_suffix : str, default=".pdf"
        File type for saving plot (PDF or PNG)

    Outputs:
    ----------
    None, plot image (PDF or PNG) saved to desired output
    folder

    By Brooke DiGia
    """
    data = np.zeros((len(sim_dirs), nwvl))
    # Loop through each simulation directory
    for i in range(len(sim_dirs)):
        metrics_output = sim_dirs[i][:-1] + "_sim_results.txt"
        print(f"Grabbing results from {sim_dirs[i]}...")

        # Calculate Strehl + metrics for ith MAOS simulation
        strehl_array, fwhm_array, rmswfe_array = calc_strehl(sim_dirs[i], metrics_output,
                                                             skysub=False, apersize=0.6)

        # Store metrics
        if (metric == "Strehl") or (metric == "strehl"):
            data[i] = np.array(strehl_array)
        elif (metric == "FWHM") or (metric == "fwhm"):
            data[i] = np.array(fwhm_array)
        elif (metric == "RMS WFE") or (metric == "RMSWFE") or \
             (metric == "rms wfe") or (metric == "rmswfe"):
            data[i] = np.array(rmswfe_array)
        else:
            print("Error: Invalid metric entered. Must be Strehl, FWHM, or RMS WFE")
            return
    
    # Average over columns of 2D data (each column corresponds to a sim wvl)
    maos_metrics = np.mean(data, axis=0)
    maos_stddevs = np.std(data, axis=0) 
    print(maos_metrics)
    print(maos_stddevs)

    # Plotting
    wvls = [0.80, 1.00, 1.25, 1.65, 2.12]
    filename = f"MAOS_{metric}_WVL_PLOT"
    fig, ax = plt.subplots()
    ax.set_xlabel(f"MAOS {metric} (calculated via PAARTI)")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"MAOS {metric} as a function of wavelength")
    ax.errorbar(wvls, maos_metrics, yerr=maos_stddevs, fmt="bo", ecolor="black", 
                capsize=4.0, capthick=2.0)
    plt.savefig(saveto + filename + file_suffix)
    return

def maos_spreadsheet_lookup(filename:str, frames:list=None, dates:list=None, *args):
    """
    Function to return sky + simulation information in numpy 2D array.
    Due to the identical nature of some of the frame names between 
    observing nights (a.k.a epochs), the dates along with the associated
    frame names are required. The output data array will include at least 
    the frame, date, and exposure time information.

    Inputs:
    --------
    filename : str
        Path to CSV spreadsheet to read into pandas dataframe object

    frames   : array, 1D variable length, dtype=str
        Array of sky frames to return names + dates + exptimes + info for

    dates    : array, 1D variable length, dtype=str
        Array of dates corresponding to above names (rquired to uniquely
        determine which on-sky frame is being referenced)
    
    *args    : optional arguments, variable length, dtype=variable
        Valid arguments include one of any combination (except None)
        of the following:
           - exptimes - exposure times in UT
           - dimm - DIMM seeing ('')
           - mass - MASS seeing ('')
           - windspd - ground layer wind speed (m/s)
           - winddir - ground layer wind direction (deg)
           - skystrehl - Sky observed Strehl
           - maosstrehl - MAOS simulation Strehl (counterpart to skystrehl)
           - maosstrehlstd - MAOS simulation Strehl standard deviation 
           - stdpercentstrehl - MAOS Strehl standard deviation as percentage
           - skyfwhm - Sky observed FWHM (mas)
           - maosfwhm - MAOS simulation FWHM
           - maosfwhmstd - MAOS FWHM standard deviation
           - stdpercentfwhm - MAOS FWHM stddev as percentage
           - skyrmswfe - Sky observed RMS WFE (nm)
           - maosrmswfe - MAOS simulation RMS WFE
           - maosrmswfestd - MAOS RMS WFE stddev
           - stdpercentrmswfe - MAOS RMS WFE stddev as percentage
           - airmass - Airmass
    
    *** To request all of the data, leave all input arguments empty save for the first (filename)
 
    Outputs:
    --------
    df       : pandas dataframe, variable shape, dtype=variable
        Pandas dataframe with columns labelled ['frames', 'date', 'exptime', '*args']
        and rows names ['frames', 'c<frame#>']    

    By Brooke DiGia
    """
    # All labels/metrics
    labels = ['frames', 'dates', 'exptimes', 'dimm', 'mass', 'windspd', 'winddir', 
              'skystrehl', 'maosstrehl', 'maosstrehlstd', 'stdpercentstrehl', 
              'skyfwhm', 'maosfwhm', 'maosfwhmstd', 'stdpercentfwhm', 'skyrmswfe', 
              'maosrmswfe', 'maosrmswfestd', 'stdpercentrmswfe', 'airmass']

    # Read in full spreadsheet
    full_table = pd.read_csv(filename, names=labels)
    
    # Data table will include frames + dates + exptimes + args ...
    datatable = [full_table[key] for key in ('frames', 'dates', 'exptimes')]
    
    if ((frames == None) and (dates != None)) or ((frames != None) and (dates == None)):
        raise ValueError("Either frames or dates is empty; ensure that the input frames and input dates correspond exactly to one another")
    elif (frames == None) and (dates == None) and (args == None):
        # User wants all data
        print(f"+ including all information...")
        return full_table
    elif (frames == None) and (dates == None) and (args != None):
        raise ValueError(f"Metric arguments {args} specified but no corresponding frames + dates")
    else:
        if args != None:
            for arg in args:
                print(f"+ include {arg} information...")
                datatable.append(full_table[arg])
        else:
            print("No additional information beyond frames, dates, and exposure times requested.")
        
        df = pd.DataFrame(np.transpose(np.array(datatable))[1:], 
                          columns=['frames', 'dates', 'exptimes', *args])
    
    return df

def fetch_sky_frames(seeds:list, skyroot:Path, baseroot:Path, *simtypes:str, dates:list=None, 
                     savecsvto:Path=None, update_maos:bool=False, verbose:bool=False):
    """
    Function to return the on-sky frames and their associated information
    available for an input observing night(s).

    Inputs:
    --------
    seeds        : array, 1D variable length, dtype=int
        Array of simulation seeds

    skyroot      : Path
        Path object for directory where dated observation sub-directories (e.g. epochs)
        are stored
    
    baseroot     : Path
       Path object for directory from which MAOS sims are run (e.g. MAOS /base/)

    dates        : array, 1D variable length, dtype=str
        Dates for which to pull on-sky observing frames. If dates is None, all
        available on-sky frame names are returned

    savecsvto    : string, default=None
        If savecsvto is None, a csv file is not of the pandas Dataframe is not
        saved. If savecsvto is entered, it should be the directory in which
        to save the csv file, named df_as_csv

        Don't forget the trailing slash!

    verbose      : boolean, default=False
        Option to include verbose output from estimate_on_sky_conditions
        subroutine    

    update_maos  : boolean, default=False
        Option to run MAOS sims for any that are missing for existing
        on-sky frames (e.g. incomplete sky-sim pairs). Otherwise
        dataframe will be returned with NaNs for any missing sims.

    simtypes     : variable, dtype=str
        Types of MAOS simulations to run, one of the following:
            - 'piston'          : 1 MAOS sim includes just piston map in NCPA parameter surf = [...]
                                  2 NCPA surf seen by wave-front sensor (SURFEVL = 1, SURFWFS = 1)
                                  3 PSD parameter empty (no windshake-jitter power spectral density 
                                    FITS input)
            - 'psd+ncpa-seen'   : 1 surf includes piston map and Kolmogorov turbulence
                                    phase screen
                                  2 PSD file included
                                  3 NCPA surf seen by wave-front sensor (SURFEVL = 1, SURFWFS = 1)
            - 'psd+ncpa-unseen' : 1 surf includes piston map and Kolmogorov turbulence phase 
                                    screen
                                  2 PSD file included
                                  3 NCPA surf NOT seen by wave-front sensor (SURFWFS = 0)

    Outputs:
    --------
    df           : pandas Dataframe, dtype=variable
        Pandas Dataframe structure containing the sky frames + info for the 
        nights specified (or all)

    By Brooke DiGia
    """  
    if dates != None:
        # User wants select sky frames + info
        names = []
        datecol = []
        sky_paths = []
        for date in dates:
            name = [f.as_posix()[-14:-9] for f in skyroot.glob(f"{date}nirc2_kp/*_psf.fits")]
            paths = [f.as_posix() for f in skyroot.glob(f"{date}nirc2_kp/*_psf.fits")]
            names.extend(name)
            temp = np.full(len(name), date)
            datecol.extend(temp.tolist())
            sky_paths.extend(paths)

        names = np.array(names)
        datecol = np.array(datecol)
        namesanddates = np.column_stack((names, datecol))
    else:
        # User wants all on-sky frames
        dates = [f.as_posix()[-16:-8] for f in skyroot.glob("*/")]
        names = []
        datecol = []
        sky_paths = []
        for date in dates:
            name = [f.as_posix()[-14:-9] for f in skyroot.glob(f"{date}nirc2_kp/*_psf.fits")]
            paths = [f.as_posix() for f in skyroot.glob(f"{date}nirc2_kp/*_psf.fits")]
            names.extend(name)
            temp = np.full(len(name), date)
            datecol.extend(temp.tolist())
            sky_paths.extend(paths)
        
        names = np.array(names)
        datecol = np.array(datecol)    
        namesanddates = np.column_stack((names, datecol))
    
    # Initialize data storage
    expstarts = []
    expstops = []
    mjds = np.empty(namesanddates.shape[0])
    frieds = np.empty(namesanddates.shape[0])
    dimms = np.empty(namesanddates.shape[0])
    masses = np.empty(namesanddates.shape[0])
    spds = np.empty(namesanddates.shape[0])
    drcts = np.empty(namesanddates.shape[0])
    airmasses = np.empty(namesanddates.shape[0])
    # MASS profile weights taken at the following atm.ht = [0 500 1000 2000 4000 8000 16000]
    masswts0 = np.empty(namesanddates.shape[0])
    masswts500 = np.empty(namesanddates.shape[0])
    masswts1000 = np.empty(namesanddates.shape[0])
    masswts2000 = np.empty(namesanddates.shape[0])
    masswts4000 = np.empty(namesanddates.shape[0])
    masswts8000 = np.empty(namesanddates.shape[0])
    masswts16000 = np.empty(namesanddates.shape[0])
    dimmtimes = []
    masstimes = []
    tubetemps = np.empty(namesanddates.shape[0])
    lgsrmswfes = np.empty(namesanddates.shape[0])
    lbwfsfwhms = np.empty(namesanddates.shape[0])
    tau0s = np.empty(namesanddates.shape[0])
    theta0s = np.empty(namesanddates.shape[0])
    sigmaDMs = np.empty(namesanddates.shape[0])
    for i, sky in enumerate(namesanddates):
        sky_file = skyroot.as_posix() + f"/{sky[1]}nirc2_kp/{sky[0]}_psf.fits"
        sky_folder = skyroot.as_posix() + f"/{sky[1]}nirc2_kp/"
       
        # Remove Strehl coordinate keywords if present
        try:
            remove_keywords(sky_file, 'XSTREHL', 'YSTREHL')
        except:
            pass

        # Grab sky FITS file header
        with fits.open(sky_file) as fits_file:
            hdu = fits_file[0]
            hdr = hdu.header

        expstarts.append(hdr['EXPSTART'])
        expstops.append(hdr['EXPSTOP'])
        mjds[i] = hdr['MJD-OBS']
        airmasses[i] = hdr['AIRMASS']
        tubetemps[i] = hdr['TUBETEMP'] 
        lgsrmswfes[i] = hdr['LGRMSWF']
        lbwfsfwhms[i] = hdr['AOLBFWHM']
 
        # Pull atm/weather info for sky file
        fried, turbpro, windspds, winddrcts, dimm, mass, dimmtime, masstime, tau_0, theta_0, sigma_DM = estimate_on_sky_conditions(sky_file, 
                                                                                                                                   sky_folder, 
                                                                                                                                   verbose)
        dimms[i] = dimm
        masses[i] = mass
        masswts0[i] = turbpro[0]
        masswts500[i] = turbpro[1]
        masswts1000[i] = turbpro[2]
        masswts2000[i] = turbpro[3]
        masswts4000[i] = turbpro[4]
        masswts8000[i] = turbpro[5]
        masswts16000[i] = turbpro[6]
        dimmtimes.append(dimmtime)
        masstimes.append(masstime)
        # Store only ground layer speed and direction quantities
        spds[i] = windspds[0]
        drcts[i] = winddrcts[0]
        # Astropy units attached to these, store only values
        tau0s[i] = tau_0.value
        theta0s[i] = theta_0.value
        frieds[i] = fried.value
        sigmaDMs[i] = sigma_DM.value
   
    # Compute metrics for on-sky frames
    sky_strehls, sky_fwhms, sky_rmswfes, sky_emp_fwhms = calc_strehl_on_sky(sky_paths, 
                                                                            "temp.txt")
    # For one type of simulation, use the collect_maos_results function to see if
    # there are on-sky observations for which MAOS sims have not been run. Note:
    # This assumes that if a MAOS sim is missing for an on-sky obs for one type
    # of simulation, it is likewise for all other input types, and all types 
    # will be run. To do for Brooke: Change this feature when possible.
    # Arbitrarily choose first simulation type entered.
    for i, type in enumerate(simtypes):
        maos_strehls, maos_strehl_stds, maos_fwhms, maos_fwhm_stds, maos_rmswfes, maos_rmswfe_stds, tot_maos_wfes, ho_maos_wfes, tt_maos_wfes, maos_emp_fwhms, maos_emp_fwhm_stds = collect_maos_results(seeds, namesanddates, baseroot, type)

    # Run missing MAOS depending on which type of simulation
    missing = np.isnan(maos_strehls)
    if verbose:
        print(f"The following frame-epoch combination of {simtypes} simulations, total of {len(missing)}, should be run: ")
        print(namesanddates[missing])

    if update_maos:
        for i, type in enumerate(simtypes):
            if verbose:
                print(f"Running {type} MAOS simulations")
            run_maos_comp_to_sky_sim(seeds, skyroot, namesanddates[missing], type, baseroot)

    # Grab newly-calculated MAOS results and store them this time
    maos_strehls_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_fwhms_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_emp_fwhms_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_rmswfes_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    tot_maos_wfes_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    ho_maos_wfes_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    tt_maos_wfes_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_stddev_strehls_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_stddev_fwhms_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_stddev_emp_fwhms_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    maos_stddev_rmswfes_alltypes = np.empty((len(sky_strehls), len(simtypes)))
    for i, type in enumerate(simtypes):
        maos_strehls, maos_strehl_stds, maos_fwhms, maos_fwhm_stds, maos_rmswfes, maos_rmswfe_stds, tot_maos_wfes, ho_maos_wfes, tt_maos_wfes, maos_emp_fwhms, maos_emp_fwhm_stds = collect_maos_results(seeds, namesanddates, baseroot, type)
        maos_strehls_alltypes[:,i] = maos_strehls
        maos_fwhms_alltypes[:,i] = maos_fwhms
        maos_emp_fwhms_alltypes[:,i] = maos_emp_fwhms
        maos_rmswfes_alltypes[:,i] = maos_rmswfes
        tot_maos_wfes_alltypes[:,i] = tot_maos_wfes
        ho_maos_wfes_alltypes[:,i] = ho_maos_wfes
        tt_maos_wfes_alltypes[:,i] = tt_maos_wfes
        maos_stddev_fwhms_alltypes[:,i] = maos_fwhm_stds
        maos_stddev_rmswfes_alltypes[:,i] = maos_rmswfe_stds
        maos_stddev_strehls_alltypes[:,i] = maos_strehl_stds
        maos_stddev_emp_fwhms_alltypes[:,i] = maos_emp_fwhm_stds

    # See if telemetry exists for on-sky dates
    _, telem_status = find_on_sky_telemetry_file(datecol, 'LGS')

    out = np.column_stack((namesanddates, mjds, telem_status, 
                           expstarts, expstops, airmasses, 
                           frieds, tau0s, theta0s,
                           dimms, dimmtimes, 
                           masses, masstimes, 
                           masswts0, masswts500, masswts1000, masswts2000, 
                           masswts4000, masswts8000, masswts16000, 
                           spds, drcts, 
                           tubetemps,
                           sky_strehls, maos_strehls_alltypes, maos_stddev_strehls_alltypes,
                           sky_fwhms, sky_emp_fwhms, maos_fwhms_alltypes, maos_emp_fwhms_alltypes,
                           maos_stddev_fwhms_alltypes, maos_stddev_emp_fwhms_alltypes,
                           lbwfsfwhms, 
                           sky_rmswfes, maos_rmswfes_alltypes, maos_stddev_rmswfes_alltypes, lgsrmswfes,
                           tot_maos_wfes_alltypes, ho_maos_wfes_alltypes, tt_maos_wfes_alltypes,
                           sigmaDMs))
    col_list = (['frames', 'dates', 'mjd', 'telem_status', 
                 'expstarts', 'expstops', 'airmasses', 
                 'frieds', 'tau0', 'theta0',
                 'dimms', 'dimmtimes', 
                 'masses', 'masstimes', 'masswts0', 'masswts500', 
                 'masswts1000', 'masswts2000', 'masswts4000', 'masswts8000', 
                 'masswts16000', 
                 'windspds', 'winddirs', 
                 'temps', 
                 'skystrehls'] + [f'maosstrehls-{sim}' for sim in simtypes] + 
                [f'maos_stds_strehls-{sim}' for sim in simtypes] + ['skyfwhms', 'skyempfwhms']
                + [f'maosfwhms-{sim}' for sim in simtypes] + [f'maosempfwhms-{sim}' for sim in simtypes] + 
                [f'maos_stds_fwhms-{sim}' for sim in simtypes] + [f'maos_stds_emp_fwhms-{sim}' for sim in simtypes] +
                ['lbwfsfwhms', 'skyrmswfes'] + [f'maosrmswfes-{sim}' for sim in simtypes]
                + [f'maos_stds_rmswfes-{sim}' for sim in simtypes] + ['lgsrmswfes']
                + [f'totmaoswfe-{sim}' for sim in simtypes] + [f'homaoswfe-{sim}' for sim in simtypes]
                + [f'ttmaoswfe-{sim}' for sim in simtypes] + ['DM_fitting_errors'])
    df = pd.DataFrame(np.array(out), columns=col_list)
    # User wants to save csv file
    if savecsvto != None:
        # Column names that are a bit more descriptive than keywords
        # aliases = ['Frame', 'Date (UT)', 'MJD', 'Telemetry?', 'Expstart (UT)', 'Expstop (UT)', 'Airmass', 'Fried (m)', 'Tau0 (s)', 'Theta0 (\'\')'
        #            'DIMM (\'\')', 'Time of DIMM (HH:MM:SS) (UT)', 'MASS (\'\')', 'Time of MASS (HH:MM:SS) (UT)', 
        #            'MASS wt 0 m', 'MASS wt 500 m', 'MASS wt 1000 m', 'MASS wt 2000 m', 'MASS wt 4000 m', 
        #            'MASS wt 8000 m', 'MASS wt 16000 m', 'Wind spd (m/s)', 'Wind drct (deg)', 'Tube Temp (Celsius)', 'Sky Strehl', 'MAOS Strehl', 'MAOS Strehl Stddev', 
        #            'Sky FWHM (mas)', 'MAOS FWHM (mas)', 'MAOS FWHM Stddev', 
        #            'Telemetry LBWFS Avg FWHM (as)', 'Sky RMS WFE (nm)', 'MAOS RMS WFE (nm)', 'MAOS RMS WFE Stddev',
        #            'Telemetry HO RMS WFE (nm)', 
        #            'MAOS-computed Total WFE (nm)', 'MAOS-computed HO WFE (nm)', 'MAOS-computed TT WFE (nm), 'DM fitting error (nm)']
        df.to_csv(savecsvto, index=False, header=col_list)
    return df

def find_on_sky_telemetry_file(dates:list, telem_type:str, 
                               telem_home:Path=Path('/g/lu/data/keck_telemetry/')):
    """
    Function to search keck_telemetry directory and see if a telemetry file exists for a night
    of observation

    Inputs:
    --------
    dates       : array-like, dtype=str
        List of on-sky observation dates for which to see if telemetry exists

    telem_type  : str
        Type of telemetry file for which to search (see VALID_TYPE below) 
        
    telem_home  : str, default='/g/lu/data/keck_telemetry/'
        Path to home of all telemetry files in which to search. Default is keck_telemetry
        location

    Outputs:
    --------
    telem_paths : list, dtype=str
        List of paths to telemetry files that exist for input on-sky observation dates

    telem_mask  : np.array, dtype=bool
        Boolean mask for use in dataframe analysis

    By Brooke DiGia
    """
    # Valid telemetry types for which to search (based on the types I have seen in keck_telemetry, will
    # expand if telemetry is sourced from another location)
    VALID_TYPE = {'LGS', 'NGS', 'fullLGS', 'fullNGS'}
    if telem_type not in VALID_TYPE:
        raise ValueError(f"find_on_sky_telemetry_file: telem_type must be one of {VALID_TYPE}")
    
    telem_paths = []
    telem_mask = np.empty(len(dates), dtype=bool)
    for i, date in enumerate(dates):
        paths = [f.as_posix() for f in telem_home.glob(f"{date}/sdata90*/nirc*/*/n*_{telem_type}_trs.sav")]
        telem_paths.extend(paths)
        if paths == []:
            # No telemetry exists for this observation date
            telem_mask[i] = False
        else:
            # Telemetry exists for this observation date
            telem_mask[i] = True

    return telem_paths, telem_mask

def run_maos_comp_to_sky_sim(seeds:list, skyroot:Path, framedates:list, simtype:str, 
                             baseroot:Path):
    """
    Function to run MAOS simulation(s) with config set by a session
    of on-sky observation (e.g. a simulation to compare to a night
    of actual on-sky observing)

    Inputs:
    --------
    framedates   : array, variable rows x 2 columns, dtype=str
        Array of on-sky frames and their correspponding epochs/dates

    seeds        : array, 1D variable length, dtype=int
        Simulation seeds - MAOS sims are run for each of these seeds
        and the results are averaged together for one on-sky frame
        MAOS counterpart result

    skyroot      : Path
        Path object for directory where dated observation sub-directories (e.g. epochs)
        are stored

    simtype      : string
       Type of simulation to run (see fetch_sky_frames header for info)

    baseroot     : Path
       Path object for directory from which MAOS sims are run (e.g. MAOS /base/)

    Outputs:
    --------
    
    By Brooke DiGia
    """  
    # Loop over the sky frames in the dataframe and run MAOS sim for each
    for i, sky in enumerate(framedates):
        sky_file = skyroot.as_posix() + f"/{sky[1]}nirc2_kp/{sky[0]}_psf.fits"
        sky_folder = skyroot.as_posix() + f"/{sky[1]}nirc2_kp/"
        
        with fits.open(sky_file) as fits_file:
            hdu = fits_file[0]
            hdr = hdu.header

        # Zenith angle
        angle = np.degrees(np.arccos(1.0/float(hdr['AIRMASS'])))
        # Wavelength at which to run MAOS (microns)
        wvl = float(hdr['TARGWAVE']) * 1.0e-6 # multiply by 1e-6 to convert from microns to m
        # STRAP WFS integration time (milli-sec)
        if 'STINTTIM' not in hdr:
            continue
        hdr_strap_int_time = float(hdr['STINTTIM'])
        # SHWFS frame rate (Hz)
        hdr_shwfs_frame_rate = float(hdr['WSFRRT'])
        hdr_strap_frame_rate = (1.0/hdr_strap_int_time)*1000.0

        # Set sim_dt to fastest WFS readout rate in Hz
        if hdr_shwfs_frame_rate > hdr_strap_frame_rate:
            max_frame_rate = hdr_shwfs_frame_rate
        else:
            max_frame_rate = hdr_strap_frame_rate

        sim_dt = 1.0/max_frame_rate
        hdr_shwfs_int_time = (1.0/hdr_shwfs_frame_rate)*1000.0 # ms
        # sim_dt = (1.0/472.0)*1000.0 # ms

        howfs_dtrat = (sim_dt / hdr_shwfs_int_time)*1000.0
        strap_dtrat = (sim_dt / hdr_strap_int_time)*1000.0
        # Compose powfs.dtrat array for input into MAOS config command override
        if howfs_dtrat < 1:
            howfs_dtrat = 1
        elif strap_dtrat < 1:
            strap_dtrat = 1
        dtrat = [int(howfs_dtrat), int(strap_dtrat), 7080]

        # Calculate siglev/bkgrnd/nearecon config parameters using variable
        # integration times from headers
        _, howfs_nearecon, howfs_siglev, howfs_bkgrnd = keck_nea_photons(8.1, 'LGSWFS', 
                                                                         hdr_shwfs_int_time/1000.0)
        _, strap_nearecon, strap_siglev, strap_bkgrnd = keck_nea_photons(14.0, 'STRAP', 
                                                                         hdr_strap_int_time/1000.0)
        nearecon = [howfs_nearecon, strap_nearecon, 8.4]
        siglev = [howfs_siglev, strap_siglev, 3723]
        bkgrnd = [howfs_bkgrnd, strap_bkgrnd, 25.3]
        # Calculate atm parameters
        fried, turbpro, windspds, winddrcts, _, _, _, _, _, _ = estimate_on_sky_conditions(sky_file, 
                                                                                           sky_folder)
        # Size of on-sky image
        # if int(hdr['NAXIS1']) == int(hdr['NAXIS2']):
        #     size = int(hdr['NAXIS1'])
        # else:
        #     size = min(int(hdr['NAXIS1']), int(hdr['NAXIS2']))

        # MAOS seems to give weird warnings when input evl.psfsize parameter is odd
        # if size % 2 != 0:
        #     size -= 1

        size = 256
        
        mode = ''
        if simtype == 'piston':
            mode = 'piston'
            surf_cmd = ["Keck_ncpa_rmswfe130nm.fits"]
            # Fetch name of current input PSD FITS file in MAOS config file keck_sim.conf
            psd_file = ''
        elif simtype == 'psd+ncpa-seen':
            mode = 'surf_wfs1'
            surf_cmd = ["Keck_ncpa_rmswfe130nm.fits", "'r0=0.36;l0=3.39;ht=40000;slope=-2; SURFWFS=1; SURFEVL=1; seed=10;'"]
            psd_file = "PSD_Keck_ws26.47mas_vib26mas_rad2.fits"
        elif simtype == 'psd+ncpa-unseen':
            mode = 'surf_wfs0'
            surf_cmd = ["Keck_ncpa_rmswfe130nm.fits", "'r0=0.36;l0=3.39;ht=40000;slope=-2; SURFWFS=0; SURFEVL=1; seed=10;'"]
            psd_file = "PSD_Keck_ws26.47mas_vib26mas_rad2.fits"
        else:
            raise ValueError(f"Invalid MAOS simulation type '{type}'. Valid types are currently: 'piston', 'psd+ncpa-seen', 'psd+ncpa-unseen'. See help() for further info")

        for seed in seeds:
            # Must be in MAOS simulation directory to run successfully
            if os.getcwd() != baseroot.as_posix():
                os.chdir(baseroot)

            maos_cmd = f"maos -o A_keck_scao_lgs_gc_{mode}_comp_{sky[0]}_seed{seed}_epoch{sky[1]} -c A_keck_scao_lgs_gc.conf evl.psfsize={size} sim.seeds={seed} evl.wvl={wvl} powfs.dtrat={dtrat} sim.zadeg={angle} powfs.siglev={siglev} powfs.bkgrnd={bkgrnd} powfs.nearecon={nearecon} sim.wspsd={psd_file} atm.r0z={fried.value} atm.wt={turbpro} atm.ws={windspds} atm.wddeg={winddrcts} surf={surf_cmd} -O"
            os.system(maos_cmd)

def collect_maos_results(seeds:list, framedates:list, baseroot:Path, simtype:str):
    """
    Function to collect existing MAOS results for input simulation seeds
    and specified on-sky counterparts and dates. Does not run any new MAOS
    simulations

    Inputs:
    --------
    seeds      : array, 1D variable length, dtype=int
        Array of simulation seeds

    framedates : array, variable rows x 2 columns, dtype=str
        Array of on-sky frames and their correspponding epochs/dates

    baseroot     : Path
       Path object for directory from which MAOS sims are run (e.g. MAOS /base/)

    simtype      : string
       Type of simulation to run (see fetch_sky_frames header for info)

    Outputs:
    --------
    strehls    : array of tuples, 1D variable length, dtype=(float, float)
        First entry is Strehl averaged over MAOS results for all simulation
        seeds. Second entry is standard deviation of these individual Strehl
        measurements
    
    fwhms      : array of tuples, 1D variable length, dtype=(float, float)
        First entry is FWHM averaged over MAOS results for all simulation
        seeds. Second entry is standard deviation of these individual FWHM
        measurements

    rmswfes    : array of tuples, 1D variable length, dtype=(float, float)
        First entry is RMS WFE averaged over MAOS results for all simulation
        seeds. Second entry is standard deviation of these individual RMS WFE
        measurements

    By Brooke DiGia
    """
    # To store metrics
    strehls = []
    strehl_stds = []
    fwhms = []
    fwhm_stds = []
    emp_fwhms = []
    emp_fwhm_stds = []
    rmswfes = []
    rmswfe_stds = []
    tot_maos_wfes = []
    ho_maos_wfes = []
    tt_maos_wfes = []

    for i in range(framedates.shape[0]):
        strehls_to_avg = []
        fwhms_to_avg = []
        emp_fwhms_to_avg = []
        rmswfes_to_avg = []
        tot_maos_to_avg = []
        ho_maos_to_avg = []
        tt_maos_to_avg = []
        for seed in seeds:
            # out_file = f"maos_comp_{framedates[i][0]}_epoch{framedates[i][1]}_seed{seed}_metrics.txt"
            out_file = "temp.txt"
            if simtype == 'piston':
                folder = baseroot.as_posix() + f"/A_keck_scao_lgs_gc_{simtype}_comp_{framedates[i][0]}_seed{seed}_epoch{framedates[i][1]}/"
            elif simtype == 'psd+ncpa-unseen':
                folder = baseroot.as_posix() + f"/A_keck_scao_lgs_gc_surf_wfs0_comp_{framedates[i][0]}_seed{seed}_epoch{framedates[i][1]}/"
            elif simtype == 'psd+ncpa-seen':
                folder = baseroot.as_posix() + f"/A_keck_scao_lgs_gc_surf_wfs1_comp_{framedates[i][0]}_seed{seed}_epoch{framedates[i][1]}/"

            try:
                maos_seed_strehls, maos_seed_fwhms, maos_seed_rmswfes, maos_seed_emp_fwhms = calc_strehl(folder, 
                                                                                                         out_file, 
                                                                                                         sim_seed=seed, 
                                                                                                         apersize=0.6) # 0.6 arcsec aperture size from spot check analysis
                _, maos_cl_metrics, _, _ = print_wfe_metrics(directory=folder, seed=seed)
                tot_maos_err = maos_cl_metrics[0]
                tt_maos_err = maos_cl_metrics[1]
                ho_maos_err = maos_cl_metrics[2]
            except Exception as error:
                print(f"Error calculating {framedates[i][0]}_{framedates[i][1]} metrics: {error} --> using NaN for metrics")
                maos_seed_strehls = [np.nan]
                maos_seed_fwhms = [np.nan]
                maos_seed_rmswfes = [np.nan]
                maos_seed_emp_fwhms = [np.nan]
                tot_maos_err = np.nan
                tt_maos_err = np.nan
                ho_maos_err = np.nan
            
            # Store values at 2.12 microns (last values)
            strehls_to_avg.append(maos_seed_strehls[-1])
            fwhms_to_avg.append(maos_seed_fwhms[-1])
            emp_fwhms_to_avg.append(maos_seed_emp_fwhms[-1])
            rmswfes_to_avg.append(maos_seed_rmswfes[-1])
            tot_maos_to_avg.append(tot_maos_err)
            ho_maos_to_avg.append(ho_maos_err)
            tt_maos_to_avg.append(tt_maos_err)
        strehls.append(np.mean(strehls_to_avg))
        strehl_stds.append(np.std(strehls_to_avg))
        fwhms.append(np.mean(fwhms_to_avg))
        fwhm_stds.append(np.std(fwhms_to_avg))
        emp_fwhms.append(np.mean(emp_fwhms_to_avg))
        emp_fwhm_stds.append(np.std(emp_fwhms_to_avg))
        rmswfes.append(np.mean(rmswfes_to_avg))
        rmswfe_stds.append(np.std(rmswfes_to_avg))
        tot_maos_wfes.append(np.mean(tot_maos_to_avg))
        ho_maos_wfes.append(np.mean(ho_maos_to_avg))
        tt_maos_wfes.append(np.mean(tt_maos_to_avg))

    return np.array(strehls), np.array(strehl_stds), np.array(fwhms), np.array(fwhm_stds), np.array(rmswfes), np.array(rmswfe_stds), tot_maos_wfes, ho_maos_wfes, tt_maos_wfes, np.array(emp_fwhms), np.array(emp_fwhm_stds)

def sky_plot(xkey:str, ykey:str, pf, colorkeys:dict):
    """
    Function to plot the pandas Dataframe holding the on-sky database
    corresponding to pf['xkey'] and pf['ykey']

    Inputs:
    --------
    xkey : str
        Data to plot on the x-axis, pf['xkey']

    ykey : str
        Data to plot on the y-axis, pf['ykey']

    pf   : pandas Dataframe
        Dataframe from which to pull data to plot

    colorkeys : dictionary, size corresponding to length of unique pf['dates']
        Color-coding between night of obs and named matplotlib color

    Outputs:
    --------
    Plots 

    By Brooke DiGia
    """
    # Ensure float datatypes prior to plotting
    pf[xkey] = pd.to_numeric(pf[xkey])
    pf[ykey] = pd.to_numeric(pf[ykey])

    # Scatter plot, color-coded by night of observation ('dates')
    color_list = [colorkeys[key] for key in pf['dates']]
    plot = pf.plot.scatter(x=xkey, y=ykey, c=plt.cm.jet, grid=True)

    # Legend information
    handles = []
    for key in colorkeys:
        patch = mpatches.Patch(color=colorkeys[key], label=key)
        handles.append(patch)
    
    plot.legend(handles=handles, ncol=2, loc='best', fontsize=10)
    plot.set_title(f'{ykey} as a function of {xkey}', fontsize=10)
    plot.set_xlabel(f'{xkey}', fontsize=8)
    plot.set_ylabel(f'{ykey}', fontsize=8)

def get_parameter_from_done_conf(directory:str, param_name:str):
    """
    Get the parameter value from a maos_done.conf file
    in the specified directory. This is a convenience function
    to fetch (or grep) the parameter value that was actually in
    the run (rather than relying on the input config files
    which might accidentally change).

    Parameters
    ----------
    directory : str
        Name of the directory (ending with /) where the
        MAOS output is stored. Routine reads
        <directory>/maos_done.conf.

    param_name : str
        Name of the parameter to search for. This should be a
        full parameter name such as fit.thetax or powfs.rne.

    Returns
    -------
    param_value : type is dynamic
        Return the parameter value as a numpy array,
        float, int, or string.
    
    """
    import re
    import sys

    file = open(f'{directory}maos_done.conf', "r")

    for line in file:
        if re.search(param_name, line):
            # Pull out just the value side (after =)
            pval = line.split("=")[1]

            # Figure out if this is an arrayed quantity. If so,
            # pull out just the stuff between [ and ] and make it an array.
            if "[" in pval:
                # Trim [ ]
                pval = pval.split("[")[1].split("]")[0]

                tmp = pval.split()
                if tmp[0].isdigit():
                    param_value = np.fromstring(pval, dtype='int', sep=' ')
                else:
                    param_value = np.fromstring(pval, dtype='float', sep=' ')
            else:
                # is int?
                if pval.isdigit():
                    param_value = int(value)
                else:
                    try:
                        # is float?
                        param_value = float(value)
                    except:
                        # default to string
                        param_value = pval

            file.close()
            return param_value

    file.close()
    return None

def centroid_residual_to_RMSWFE(telem_path:str):
    """
    Function to calculate RMS wave front error from centroid offsets per subaperture from
    telemetry file

    Inputs:
    --------
    telem_path : str
        Path to telemetry file

    Outputs:
    --------
    phibar     : float
        Average RMS WFE (nm)

    phi        : array, dtype=float
        RMS WFE (nm) averaged over actuators as a function of telemetry time

    data.a.residualrms[0] : array, dtype=float, length=duration of telemetry timestream
        Residual RMS measurements spaced ~ 1 ms apart (delta_t in telemetry timestamps).
        Return this alongside calculated phi for comparison

    Math
    -----
    phi_rms = sqrt [sum over all actuators(a_ci - a_0i - u)^2 / num_acutators)],
    where a_ci is the calculated movement of ith DM actuator, a_0i is the 
    ground ''truth'' of ith DM actuator, and 
    u = sum over all actuators(a_ci - a_0i) / num_actuators
    from: https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-1-301&id=544659

    By Brooke DiGia
    """
    data = load_telemetry(telem_path)

    # Before calculating, mask bad acutators
    # bad_ap, bad_act = examine_subapertures(telem_path=telem_path)
      
    # Average over actuators on DM 
    # u = np.mean(np.subtract(data.a.dmcommand[0], data.a.residualwavefront[0][:, 0:349]), axis=1)
    # u = np.tile(u, (data.a.residualwavefront[0][:, 0:349].shape[1], 1)).T
    # phi = np.sqrt( np.mean(np.square(data.a.dmcommand[0] - data.a.residualwavefront[0][:, 0:349] - u), 
    #                        axis=1) )

    # Try just a few actuators near center we know to be good by visual inspection
    # dm_movements = np.subtract(data.a.dmcommand[0][:, 50:56], data.dm_origin[50:56])
    # u = np.mean(np.subtract(data.a.residualwavefront[0][:, 50:56], dm_movements), axis=1)
    # u = np.tile(u, (data.a.residualwavefront[0][:, 50:56].shape[1], 1)).T
    # phi = np.sqrt( np.mean(np.square(data.a.residualwavefront[0][:, 50:56] - dm_movements - u), 
    #                        axis=1) )
    
    # Try direct RMS of just residualwavefront
    phi = np.sqrt(np.mean(np.square(data.a.residualwavefront[0][:, 50:56])))
    
    phi *= (0.6 * 1000.0) # 0.6 microns/volts, then * 1000.0 for microns to nm
    # Average over time
    phibar = np.mean(phi)

    return phibar, phi, data.a.residualrms[0] 

def shwfs_supapint(telem_path:str):
    """
    Function to calculate the SUPAPERTURE intensity using telemetry data for 
    304 subapertures. 

    Units of MAOS config parameter siglev : only listed as signal level at sim.dtref
    powfs.bkgrnd = sky background in unit e/pixel/frame at sim.dtref

    Inputs:
    --------
    telem_path     : str
        Path to telemetry file

    Note regarding SUBAPINTENSITY units: 
    https://mirametrics.com/help/mira_al_8/source/magnitude_calculations.htm

    The value of Counts is the net signal from the object, above the sky background, 
    and measured in the raw pixel value units (often called "ADU's"). The telemetry 
    KAON specifies SUBAPINTENSITY units as adu, so I believe they are already in counts

    Outputs:
    --------
    flux_per_subap : 2D array, dtype=float
        Array of fluxes per 304 subapertures over time ([304, length of timestream]),
        along with the mean of this array over apertures and time for input into
        MAOS config parameter powfs.siglev[0] (first entry for SHWFS quantity)

    By Brooke DiGia
    """
    data = load_telemetry(telem_path)
    shwfs_gain = 0.508 #e-/ADU
    shwfs_int_time = 0.0
    for item in data.header:
        decoded:str = item.decode('ascii')
        # SHWFS frame rate (Hz)
        if decoded.startswith('WSFRRT'):
            frame_rate = decoded.split(' ')[3]
            shwfs_int_time = (1.0/float(frame_rate[1:])) # seconds

    # SUBAPINTENSITY is dark-subtracted and flat-field corrected intensity per 304 subaperture
    # Flux = Gain * Counts / Exptime (https://mirametrics.com/help/mira_al_8/source/magnitude_calculations.htm)
    flux_per_subap = ( data.a.subapintensity[0] * shwfs_gain ) / shwfs_int_time
    return flux_per_subap, np.mean(flux_per_subap)

def strap_flux(telem_path:str):
    """
    Function to calculate 

    Inputs:
    -------
    telem_path     : str
        Path to telemetry file

    Outputs:
    --------
    data.b.apdcounts[0]           : array, dtype=float
        Array of APDCOUNTS data from telemetry in units of counts
        Compare this to calc_apdcounts

    np.mean(data.apd_sky_back[0]) : float
        For input into MAOS config parameter powfs.bkgrnd for STRAP entry (powfs.bkgrnd[1]).

    calc_apdcounts                : float
        APDCOUNTS average calculated from STRAPDQMN (only one quantity per night of telemetry)

    flux_from_calc_apdcounts      : float

    flux                          : float
        

    By Brooke DiGia
    """
    data = load_telemetry(telem_path)
    calc_apdcounts = np.empty(data.b.apdcounts[0].shape)

    strapdqmn = 0.0
    strap_int_time = 0.0
    for item in data.header:
        decoded:str = item.decode('ascii')
        if decoded.startswith('STRAPDQMN') or decoded.startswith('STAPDQMN'):
            dqmn = decoded.split(' ')[1]
            strapdqmn = float(dqmn[1:])
        # STRAP integration time (ms)
        elif decoded.startswith('STINTTIM'):
            int_time = decoded.split(' ')[1]
            strap_int_time = float(int_time[1:])

    calc_apdcounts = strapdqmn * strap_int_time
    # flux_from_calc_apdcounts = calc_apdcounts * strap_gain 
    # flux = data.b.apdcounts[0] * strap_gain 
    return data.b.apdcounts[0], calc_apdcounts, np.mean(data.apd_sky_back[0]) #, flux_from_calc_apdcounts

def load_telemetry(telem_path:str):
    """
    Function to load in telemetry file from Path location

    Inputs:
    --------
    telem_path : str
        Path to telemetry file

    Outputs:
    --------
    data       : dictionary object
        Dictionary object containing telemetry data

    By Brooke DiGia
    """
    data = readsav(telem_path)
    return data

def approximate_num_actuators(dmdx:float):
    """
    Function to approximate the number of actuators on Keck AO DM 
    based on MAOS dm.dx parameter. This approximation assumes a circular deformable
    mirror where the primary mirror is 11 m across.

    Inputs:
    --------
    dmdx           : float
        MAOS dm.dx parameter value (m)

    Outputs:
    --------
    N_act          : float
        Number of actuators calculated directly from equation

    np.ceil(N_act) : float
        Rounded number of actuators for input into MAOS

    By Brooke DiGia, calculation of number of actuators N_act from Brianna Peck
    """
    D = 11 # meters
    N_act = ( np.pi * D**2.0 ) / (4.0 * dmdx**2.0)
    print(f"Rounding {N_act} to {np.ceil(N_act)}")
    return N_act, np.ceil(N_act)

def check_symmetric(a:list, rtol:float=1e-05, atol:float=1e-08):
    """
    Function to check if input 2D list/array is symmetric.

    Inputs:
    --------
    a    : list
        List/array to check

    rtol : float, default=1e-5
        Relative tolerance parameter

    atol : float, default=1e-8
        Absolute tolerance parameter

    By Brooke DiGia    
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def load_sub_ap_map(subap_map:str="/Users/bdigia/code/python/paarti/paarti/utils/sub_ap_map.txt"):
    """
    Function to load the Keck DM subaperture map and convert it into
    a Numpy array free of whitespace or other delimiters (e.g. \n).

    Inputs:
    -------
    subap_map          : str
        Path to .txt file of sub aperture map

    Outputs:
    --------
    sub_ap_map_cleaned : np.array, dtype=int, [20, 20]
        Numpy 2D array of sub apertures (1s and 0s)

    By Brooke DiGia
    """
    sub_ap_map = open(subap_map, 'r').read().split('\n')[::-1]
    sub_ap_map = np.array(sub_ap_map[1:])
    sub_ap_map_cleaned = []
    for line in sub_ap_map:
        cleaned = line.split(' ')
        str_list = list(filter(None, cleaned))
        sub_ap_map_cleaned.append(str_list)
    
    sub_ap_map_cleaned = np.array(sub_ap_map_cleaned)
    return sub_ap_map_cleaned.astype(int)

def load_act_map(actuator_map:str="/Users/bdigia/code/python/paarti/paarti/utils/actuator_map.txt"):
    """
    Function to load the Keck DM actuator map and convert it into
    a Numpy array free of whitespace or other delimiters (e.g. \n).
    Analogous to load_sub_ap_map().

    Inputs:
    -------
    actuator_map    : str
        Path to .txt file of actuator map

    Outputs:
    --------
    act_map_cleaned : np.array, dtype=int, [21, 21]
        Numpy 2D array of actuators (1s and 0s)

    By Brooke DiGia
    """
    act_map = open(actuator_map, 'r').read().split('\n')[::-1]
    act_map = np.array(act_map)
    act_map_cleaned = []
    for line in act_map:
        cleaned = line.split(' ')
        str_list = list(filter(None, cleaned))
        act_map_cleaned.append(str_list)
    
    act_map_cleaned = np.array(act_map_cleaned)
    return act_map_cleaned.astype(int)

def examine_subapertures(telem_path:str, thres:float=0.3, visualize_dm:bool=True,
                         subap_map:str="/Users/bdigia/code/python/paarti/paarti/utils/sub_ap_map.txt", 
                         actuator_map:str="/Users/bdigia/code/python/paarti/paarti/utils/actuator_map.txt"):
    """
    Function to evaluate the sub-apertures on a given night/observation/telemetry stream
    and designate those that are partially illuminated (flux below a threshold from the median)

    Inputs:
    --------
    telem_path   : str
        Path to telemetry file

    thres        : float, default=0.3 (30%)
        Threshold % for which to designate actuators as partially illuminated 

    visualize_dm : bool, default=True
        Option to visualize the map of subapertures and actuators, marking which
        ones will be masked

    subap_map    : str
        Path to .txt map of Keck DM subapertures

    actuator_map : str
        Path to .txt map of Keck DM actuators

    Outputs:
    --------
    bad_subap    : list, dtype=float
        List of bad subapertures (in terms of subaperture indexing)

    bad_act      : 2D numpy array, dtype=int
        Map of bad actuators (actuator mask) on Keck DM

    Displays DM subapertures and actuators, colored by subaperture intensity [adu].
    Returns indices of bad subapertures and the derived bad actuators. The bad actuators
    are derived based on: 

    “The valid actuator map is the one that jointly covers the union of the four 
    valid subaperture map from each of the four LGS channels (as advocated in AD 2).”
    KAON 1320, KAPA LTAO RTC Algorithm Description, Correia, Surendran, Wizinowich, 
    Cetre, Ragland, et. al

    By Brooke DiGia
    """
    data = load_telemetry(telem_path)
    med_per_subap = np.median(data.a.subapintensity[0][:,].astype(float), axis=0)

    # Calculate subaperture intensity statistics
    subapint_std = np.std(data.a.subapintensity[0][:,].astype(float), axis=0)
    subapint_med = np.median(med_per_subap)
    subapint_mean = np.mean(data.a.subapintensity[0][:,].astype(float), axis=0)
    median_flux_timestream = np.median(data.a.subapintensity[0][:,].astype(float), axis=0)
    # print(f"Median SUBAPINTENSITY [adu] = {median_flux_timestream}")
    # print(f"Median SUBAPINTENSITY [adu] = {np.median(median_flux_timestream)}")

    # Find where time averaged intensity of each subaperture is less than
    # 30% of median value
    bad_subap = np.where(subapint_mean < thres*subapint_med)[0]

    # Load sub aperture and actuator maps for Keck DM
    subapmap = load_sub_ap_map().astype(int)
    actmap = load_act_map().astype(int)

    # Turn sub-apertures off (1 -> 0) in sub aperture map if sub aperture is selected as bad
    subap_index = -1 # (start at -1 because there is a subaperture labeled 0 itself)

    for row in range(subapmap.shape[0]-1, -1, -1):
        for col in range(subapmap.shape[1]):
            # Track subapertures with special separate index that 
            # only counts the 1s (subapertures) in the subap map
            if subapmap[row,col] == 1:
                subap_index += 1
                # With subaperture index updated, evaluate subaperture within
                # ORIGINAL subaperture map
                if subap_index in bad_subap:
                    # Turn bad subaperture 'off' by setting it to 0 (it is
                    # no longer considered a subaperture in union map below)
                    subapmap[row, col] = 0

    # Pad map with one row and one column of 0s, will be elimated during subsequent logic operations
    temp = np.c_[subapmap, np.zeros(subapmap.shape[0])]
    exp = np.r_[temp, [np.zeros(temp.shape[1])]]

    # Find which corresponding actuators need to be masked as well
    union = np.bitwise_and(actmap, exp.astype(int))

    # Trim off 0 pads post-union
    bad_act = union[:-1,:-1]

    # Display map with colorbar
    if visualize_dm:
        with open(subap_map, 'r') as sub_aperture_map:
            with open(actuator_map, 'r') as actuator_map:
                exact_x = []
                exact_y = []
                start_y = 0.1
                act_x = []
                act_y = []
                start_act_y = 0.0

                for info in sub_aperture_map.readlines():
                    start_x = 0.1
                    line = list(info)
                    for i in line:
                        if (i != " ") & (i != '\n'):
                            if int(i) != 0:
                                exact_x.append(start_x)
                                exact_y.append(start_y)
                            start_x += 0.2
                    start_y += 0.2

                for info in actuator_map.readlines():
                    start_x = 0.0
                    line = list(info)
                    for i in line:
                        if (i != " ") & (i != '\n'):
                            if int(i) != 0:
                                act_x.append(start_x)
                                act_y.append(start_act_y)
                            start_x += 0.2
                    start_act_y += 0.2

                cmap = mpl.cm.inferno
                str_txt = [str(i) for i in range(304)]
                fig, axes = plt.subplots(figsize=(17.5, 15))
                im = plt.scatter(exact_x, exact_y, label="subaperture", alpha=1.0, 
                                 c=pd.Series(subapint_mean), cmap=cmap, s=75)
                axes.set_aspect('equal')
                fig.colorbar(im, ax=axes, 
                             label='SUBAPINTENSITY (averaged over timestream) [adu]', 
                             shrink=0.7)
                i = 0
                while i < 304:
                    if i in bad_subap:
                        axes.text(exact_x[i], exact_y[i]+0.05, str_txt[i], 
                                  horizontalalignment='center', fontsize=10, color="red")
                    else:
                        axes.text(exact_x[i], exact_y[i]+0.05, str_txt[i], 
                                  horizontalalignment='center', fontsize=10, color="black")
                    i += 1

                cbar = plt.gcf()
                cbar_ax = cbar.axes[-1]
                cbar_ax.tick_params(labelsize=20)
                cbar_ax.set_frame_on(True)

                plt.scatter(act_x, act_y, color="grey", label="actuator", alpha=1.0, marker="*", s=50)
                axes.set_title("Keck DM Subapertures and Actuators, colored by subaperture intensity [adu]", 
                               fontsize=14)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    return bad_subap, bad_act
 
def telemetry_data(telem_paths:list=None):
    """
    Function to create Pandas dataframe object from input telemetry files/paths.

    Inputs:
    -------
    telem_paths : array, dtype=str, default=None
        Array of input telemetry files for which to collect telemetry into
        dataframe. If telem_paths is None (none are input), assume the user
        wants all LGS telemetry files loaded and put into dataframe

    Outputs:
    --------
    df          : Pandas dataframe, mixed data types
        Telemetry dataframe

    By Brooke DiGia
    """
    if telem_paths == None:
        # Fetch names of all 'LGS' telemetry files (takes ~few seconds)
        telem_home = Path("/g/lu/data/keck_telemetry/")
        paths = [f.as_posix() for f in telem_home.glob(f"*/sdata90*/nirc*/*/n*_LGS_trs.sav")]
    else:
        paths = telem_paths

    lgrmswfes = np.empty(len(paths))
    strapdqmns = np.empty(len(paths))
    strap_int_times = np.empty(len(paths))
    strap_time_intervals = np.empty(len(paths))
    shwfs_int_times = np.empty(len(paths))
    shwfs_time_intervals = np.empty(len(paths))
    apdcounts = np.empty(len(paths))
    apdskybkgrnds = np.empty(len(paths))
    mean_residualrms = np.empty(len(paths))
    mean_rmswfe = np.empty(len(paths))
    mean_subapint = np.empty(len(paths))
    strap_siglevs = np.empty(len(paths))
    strap_bkgrnds = np.empty(len(paths))
    shwfs_siglevs = np.empty(len(paths))
    shwfs_bkgrnds = np.empty(len(paths))
    flux_from_subapint = np.empty(len(paths))
    for i, path in enumerate(paths):
        print(f"Telemetry file {i+1} out of {len(paths)} | {path}")
        data = load_telemetry(path)
        
        # Collect header metrics for this telemetry file
        for item in data.header:
            decoded:str = item.decode('ascii')
            # LGRMSWF = closed-loop HO RMS WF residual for the night of telemetry
            if decoded.startswith('LGRMSWF'):
                lgrmswf = decoded.split(' ')[2]
                lgrmswfes[i] = float(lgrmswf[1:])
            # STRAPDQMN = strap quad mean apd counts for the night of telemetry
            # some telemetry headers having typo keyword STAPDQMN
            elif decoded.startswith('STRAPDQMN') or decoded.startswith('STAPDQMN'):
                dqmn = decoded.split(' ')[1]
                strapdqmns[i] = float(dqmn[1:])
            # STRAP integration time (ms)
            elif decoded.startswith('STINTTIM'):
                int_time = decoded.split(' ')[1]
                strap_int_times[i] = float(int_time[1:])
            # SHWFS frame rate (Hz)
            elif decoded.startswith('WSFRRT'):
                frame_rate = decoded.split(' ')[3]
                shwfs_int_times[i] = (1.0/float(frame_rate[1:]))*1000.0 # ms
            
        # APDCOUNTS data array within telemetry is [4, length of timestream]
        # or transpose ([length of timestream, 4]), where 4 is four the four
        # WFS quadrants. Average over time and space (four quadrants) to get
        # one quantity for each night
        apdcounts[i] = np.mean(data.b.apdcounts[0])
        apdskybkgrnds[i] = np.mean(data.apd_sky_back[0])

        # RESIDUALRMS data array within telemetry is one number for all actuators/subapertures
        # for each time stamp. Average over this timestream to get one quantity for each night
        # of telemetry
        mean_residualrms[i] = np.mean(data.a.residualrms[0][0])

        # SUBAPINTENSITY data array within telemetry is timestream of numbers for each of 304
        # subapertures. Average over this entire array to get one quantity for each night of
        # telemetry
        mean_subapint[i] = np.mean(data.a.subapintensity[0])
        shwfs_gain = 0.508 # e-/ADU
        flux_from_subapint[i] = np.mean(( data.a.subapintensity[0] * shwfs_gain ) / shwfs_int_times[i])

        # Calculate RMS WFE for telemetry night using RESIDUALWAVEFRONT data array
        phi, phi_t, _ = centroid_residual_to_RMSWFE(path)
        mean_rmswfe[i] = phi

        # keck_nea_photons : use header integration times to compute MAOS input config parameters
        # to compare against telemetry
        _, _, howfs_siglev, howfs_bkgrnd = keck_nea_photons(8.1, 'LGSWFS', shwfs_int_times[i]/1000.0)
        _, _, strap_siglev, strap_bkgrnd = keck_nea_photons(14.0, 'STRAP', strap_int_times[i]/1000.0)
        strap_siglevs[i] = strap_siglev 
        strap_bkgrnds[i] = strap_bkgrnd
        shwfs_siglevs[i] = howfs_siglev
        shwfs_bkgrnds[i] = howfs_bkgrnd

        # Timestamp intervals to compare to header integration times
        strap_time_intervals[i] = np.mean(np.diff(data.b.timestamp[0])) * 100.0 * (1e-9) * 1000.0 # ms
        shwfs_time_intervals[i] = np.mean(np.diff(data.a.timestamp[0])) * 100.0 * (1e-9) * 1000.0 # ms

    out = np.column_stack((paths, lgrmswfes, strapdqmns, mean_subapint, 
                           strap_int_times, strap_time_intervals,
                           shwfs_int_times, shwfs_time_intervals,
                           apdcounts, apdskybkgrnds, 
                           flux_from_subapint,
                           mean_residualrms, mean_rmswfe, 
                           strap_siglevs, strap_bkgrnds, 
                           shwfs_siglevs, shwfs_bkgrnds))
    col_list = (['filename', 'LGRMSWFs', 'STRAPDQMNs', 'SUBAPINTs [adu]', 
                 'STINTTIMs', 'DATA.B Time Interval',
                 'SHWFS int times', 'DATA.A Time Interval',
                 'APDCOUNTS', 'APD_SKY_BACK',
                 'SHWFS FLUX',
                 'RESIDUALRMS', 'RMSWFE', 
                 'STRAP SIGLEV', 'STRAP BKGRND', 
                 'SHWFS SIGLEV', 'SHWFS BKGRND'])
    df = pd.DataFrame(np.array(out), columns=col_list)
    # Column names that are a bit more descriptive than keywords
    aliases = ['Telemetry file', 'LGRMSWF (nm)', 'STRAPDQMN quad mean APD counts', 
               'SUBAPERTURE MEAN INTENSITIES [adu]', 'STRAP INT TIMEs (ms)', 
               'Spacing of STRAP telemetry (ms)', 'HO SHWFS INT TIMEs (ms)', 
               'Spacing of SHWFS telemetry (ms)', 'APDCOUNTS [adu]', 
               'APD_SKY_BACK average over quad [adu]',
               'SHWFS FLUX from SUBAPINTENSITY (e-/sec)', 
               'RESIDUALRMS (nm)', 'RMSWFE calculated from residualwavefront data [nm]', 
               'STRAP siglev knp', 'STRAP bkgrnd knp', 'SHWFS siglev knp', 
               'SHWFS bkgrnd knp']
    df.to_csv("/Users/bdigia/work/ao/keck/maos/keck/my_base/LGS_telemetry.csv", 
              index=False, header=aliases)
    
    return df

def tt_fft(tt:list, delta_t:float):
    """
    Function to take FFT of TT residual timestream (one night's/telemetry file's
    TT residual)

    Inputs:
    -------
    tt              : Numpy array, dtype=float
        TT residual in arcsec, either x or y dimension

    delta_t         : float
        Average time spacing between timestamps in telemetry timestream
        in seconds
        
    Outputs:
    --------
    freq            : Numpy array, dtype=float
        Frequencies of Fourier transform of TT residual (i.e. TT residual
        power spectrum)

    ttcentroids_fft : Numpy array, dtype=float
        TT residual power spectrum

    By Brooke DiGia
    """
    plt.rcParams.update({"text.usetex": False, 
                         "font.sans-serif": "Helvetica",})
    n = len(tt)
    # Real Fourier transform TT centroids into frequency space
    ttcentroids_fft = np.fft.rfft(tt)
    # Grab real frequencies
    freq = np.fft.rfftfreq(n, d=delta_t)
    return freq, ttcentroids_fft

def sci_format(x, lim):
    return '{:.6e}'.format(x)

def tt_residuals(telem_paths:list=None):
    """
    Function to calculate the averaged tip-tilt (TT) residual in arcsec
    for various input telemetry files (corresponding to nights
    of observation)

    Inputs:
    -------
    telem_paths        : array, dtype=str, default=None
        Array of input telemetry files for which to collect telemetry into
        dataframe. If telem_paths is None (none are input), assume the user
        wants all LGS telemetry files loaded and put into dataframe

    Outputs:
    --------
    avg_tt_centroids_x : array, len(telem_paths), dtype=float
        Array of TT x average residuals per night (telemetry file)

    avg_tt_centroids_y : array, len(telem_paths), dtype=float
        Array of TT y average residuals per night (telemetry file)

    avg_tt_laser_res   : array, len(telem_paths), dtype=float
        Array of average TT residuals as measured by LGS on SHWFS

    By Brooke DiGia
    """
    if telem_paths == None:
        # Fetch names of all 'LGS' telemetry files (takes ~few seconds)
        telem_home = Path("/g/lu/data/keck_telemetry/")
        paths = [f.as_posix() for f in telem_home.glob(f"*/sdata90*/nirc*/*/n*_LGS_trs.sav")]
    else:
        paths = telem_paths

    avg_tt_centroid_x = np.empty(len(paths))
    avg_tt_centroid_y = np.empty(len(paths))
    avg_tt_laser_res = np.empty(len(paths))
    break_freqs = np.empty(len(paths))
    welch_break_freqs = np.empty(len(paths))
    for i, path in enumerate(paths):
        print(f"Telemetry file {i+1} out of {len(paths)} | {path}")
        data = load_telemetry(path)
        print(data.b.timestamp[0][-1] - data.b.timestamp[0][0])

        # Collect original frame number (FILENAME) and date of
        # observation (DATE-OBS) for use in plot labels
        frame = ''
        dateobs = ''
        year = ''
        strapdqmn = 0.0
        for item in data.header:
            decoded:str = item.decode('ascii')
            if decoded.startswith('FILENAME'):
                frame = decoded.split(' ')[1][1:]
            elif decoded.startswith('DATE-OBS'):
                dateobs = decoded.split(' ')[1][1:]
                year = dateobs[:4]
            elif decoded.startswith('TSTAMP_STR_START'):
                startstamp = decoded.split(' ')[1]
            # STRAPDQMN = strap quad mean apd counts for the night of telemetry
            # some telemetry headers having typo keyword STAPDQMN
            elif decoded.startswith('STRAPDQMN') or decoded.startswith('STAPDQMN'):
                dqmn = decoded.split(' ')[1]
                strapdqmn = float(dqmn[1:])

        # According to KAON 1165 AO Telemetry, DTTCENTROIDS is a measure of the TT 
        # residual in arcsec
        ttcentroids_x = data.b.dttcentroids[0][:,0]
        ttcentroids_y = data.b.dttcentroids[0][:,1]
        # DTTCOMMANDS are 'down' tip-tilt actuator commands in absolute offsets
        # (arcsec)
        dtt_commands = data.b.dttcommands[0]
        dtt_commands_x = dtt_commands[:,0]
        dtt_commands_y = dtt_commands[:,1]
        # Measurement source is from laser, not from NGS on TT sensor, hence why it lives in data.a
        tt_laser_res = data.a.residualwavefront[0][:, 349:350] 
        avg_tt_centroid_x[i] = np.mean(ttcentroids_x)
        avg_tt_centroid_y[i] = np.mean(ttcentroids_y)
        avg_tt_laser_res[i] = np.mean(tt_laser_res)

        # Fourier transforms
        print(np.mean(np.diff(data.b.timestamp[0]))*100.0*(1e-9))
        dt = np.mean(np.diff(data.b.timestamp[0]))*100.0*(1e-9)
        x_freq, x_tt_fft = tt_fft(ttcentroids_x, 
                                  delta_t=dt) # delta_t in sec
        y_freq, y_tt_fft = tt_fft(ttcentroids_y, 
                                  delta_t=dt) # delta_t in sec
        # Average X and Y spectra together to reduce noise, overlay on plot
        mean_fft = np.mean([x_tt_fft, y_tt_fft], axis=0)
        freq = x_freq
        # Put this averaged FFT into noise reduction to further reduce noise
        # Savitzky-Golay filter
        mean_fft_filtered=Savitzky_Golay(mean_fft, 5, 2)
        # Convert FFT result to power spectral density (PSD)
        psd_fft = np.square(np.abs(mean_fft)) / (2*dt)
        psd_fft_filt = np.square(np.abs(mean_fft_filtered)) / (2*dt)
        # Welch's method for PSD direct from timestream signal (TT centroids)
        f, pxx_den_x = signal.welch(ttcentroids_x, nperseg=100, fs=(1/dt), scaling='density')
        _, pxx_den_y = signal.welch(ttcentroids_x, nperseg=100, fs=(1/dt), scaling='density')
        pxx_den_avg = np.mean( np.array([ pxx_den_x, pxx_den_y ]), axis=0 )
        pxx_den_sg = Savitzky_Golay(pxx_den_avg, 5, 2)

        # Pseudo open-loop calculations:
        # Add commands to centroids = offsets to create pseudo open loop (pol)
        # measurement (arrays should already be aligned and same length despite time
        # lag between entering of command and execution by actuator)
        dtt_pol_x = np.add(dtt_commands_x, ttcentroids_x)
        dtt_pol_y = np.add(dtt_commands_y, ttcentroids_y)
        # FFT
        pol_freq, dtt_pol_fftx = tt_fft(dtt_pol_x, 
                                        delta_t=dt)
        _, dtt_pol_ffty = tt_fft(dtt_pol_y, 
                                 delta_t=dt)
        mean_pol_fft = np.mean([dtt_pol_fftx, dtt_pol_ffty], axis=0)
        mean_pol_fft_filtered = Savitzky_Golay(mean_pol_fft, 5, 2)
        # Convert FFT result to power spectral density (PSD)
        psd_pol_fft = np.square(np.abs(mean_pol_fft)) / (2*dt)
        psd_pol_fft_filt = np.square(np.abs(mean_pol_fft_filtered)) / (2*dt)
        # Welch
        f, pxx_den_x_pol = signal.welch(dtt_pol_x, nperseg=100, fs=(1/dt), scaling='density')
        _, pxx_den_y_pol = signal.welch(dtt_pol_y, nperseg=100, fs=(1/dt), scaling='density')
        pxx_den_avg_pol = np.mean( np.array([ pxx_den_x_pol, pxx_den_y_pol ]), axis=0 )
        pxx_den_sg_pol = Savitzky_Golay(pxx_den_avg_pol, 5, 2)

        # Find where break in PSD occur (up to what frequency is Keck AO
        # correcting tip-tilt?)
        break_freqs[i] = find_tt_break(psd_fft_filt, 
                                       psd_pol_fft_filt, 
                                       freq=freq)
        welch_break_freqs[i] = find_tt_break(pxx_den_sg, 
                                             pxx_den_sg_pol, 
                                             freq=f)
        
        # Check for telemetry decimation
        _, _, _, _, strap_decimation_bool, shwfs_decimation_bool = telemetry_decimation(data)
        print(f"Is STRAP telemetry data decimated? {strap_decimation_bool}")
        print(f"Is SHWFS telemetry decimated? {shwfs_decimation_bool}")

        if (i < 10) or (i % 50 == 0):
            plt.rcParams.update({"text.usetex": False, "font.sans-serif": "Helvetica"})
            plt.rc('legend', fontsize=12)
            fig = plt.figure(figsize=(12.5, 15.0), layout='constrained')
            spec = fig.add_gridspec(5, 2)
            if strap_decimation_bool and shwfs_decimation_bool:
                fig.suptitle(f"Tip-tilt (TT) Residuals ('') on {dateobs} in {frame} | Telemetry decimated")
            elif strap_decimation_bool and (not shwfs_decimation_bool):
                fig.suptitle(f"Tip-tilt (TT) Residuals ('') on {dateobs} in {frame} | STRAP telemetry decimated")
            elif shwfs_decimation_bool and (not strap_decimation_bool):
                fig.suptitle(f"Tip-tilt (TT) Residuals ('') on {dateobs} in {frame} | STRAP telemetry decimated")
            else:
                fig.suptitle(f"Tip-tilt (TT) Residuals ('') on {dateobs} in {frame} | No decimation")
            ax0 = fig.add_subplot(spec[0,0])
            ax1 = fig.add_subplot(spec[0,1])
            ax2 = fig.add_subplot(spec[1, :])
            ax3 = fig.add_subplot(spec[2, :])
            ax4 = fig.add_subplot(spec[3, :])
            ax5 = fig.add_subplot(spec[4, :])
            ax0.grid(True, linestyle='dotted')
            ax1.grid(True, linestyle='dotted')
            ax2.grid(True, linestyle='dotted')
            ax3.grid(True, linestyle='dotted')
            ax4.grid(True, linestyle='dotted')
            ax5.grid(True, linestyle='dotted')

            # TT residuals in X
            ax0.plot(data.b.timestamp[0] / (1e9), ttcentroids_x, 'k.-', linewidth=0.5, alpha=0.2)
            ax0.axhline(y=np.mean(ttcentroids_x), color='cyan', linestyle='--', linewidth=2, 
                        label=r"$\bar{\mathtt{residual x tt}}$ arcsec")
            ax0.text(ax0.get_xlim()[0], np.mean(ttcentroids_x)*1.03, 
                     s=f"{np.mean(ttcentroids_x):.3f}", color='cyan')
            ax0.set_title("X")
            ax0.xaxis.set_major_formatter(FuncFormatter(sci_format))
            try:
                ax0.set_xlabel(f"Time (sec) from beginning of {year} ({startstamp})")
            except UnboundLocalError:
                ax0.set_xlabel(f"Time (sec) from beginning of {year}")
            for label in ax0.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

            # TT residuals in Y
            ax1.plot(data.b.timestamp[0] / (1e9), ttcentroids_y, 'k.-', linewidth=0.5, alpha=0.2)
            ax1.axhline(y=np.mean(ttcentroids_y), color='magenta', linestyle='--', 
                        linewidth=2, label=r"$\bar{\mathtt{residual y tt}}$ arcsec")
            ax1.text(ax1.get_xlim()[0], np.mean(ttcentroids_y)*1.03, 
                     s=f"{np.mean(ttcentroids_y):.3f}", color='magenta')
            ax1.xaxis.set_major_formatter(FuncFormatter(sci_format))
            try:
                ax1.set_xlabel(f"Time (sec) from beginning of {year} ({startstamp})")
            except UnboundLocalError:
                ax1.set_xlabel(f"Time (sec) from beginning of {year}")
            ax1.set_title("Y")
            for label in ax1.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

            # PSDs
            # axes[2].semilogy(x_freq, np.abs(x_tt_fft), linewidth=0.3, color='cyan', 
            #                  label="$\mathtt{FFT}(TT_{x, CL})$")
            # axes[2].semilogy(y_freq, np.abs(y_tt_fft), linewidth=0.3, color='magenta', 
            #                  label="$\mathtt{FFT}(TT_{y, CL})$")
            ax2.semilogy(freq, psd_fft, linewidth=0.3, color="gold", 
                         label="Mean of $\mathtt{PSD}(TT_{x, CL})$ & $\mathtt{PSD}(TT_{y, CL})$")
            ax2.semilogy(freq, psd_fft_filt, linewidth=0.5, color="grey", 
                         label="Savitzsky-Golay-filtered PSD")
            ax2.set_title("Residual closed-loop tip-tilt power spectral density (PSD)")
            ax2.legend(ncol=2)
            ax2.set_ylabel("Log PSD")

            # axes[3].semilogy(pol_freq, np.abs(dtt_pol_fftx), linewidth=0.3, color="indigo", 
            #                  label="$\mathtt{FFT}(TT_{x, POL})$")
            # axes[3].semilogy(pol_freq, np.abs(dtt_pol_ffty), linewidth=0.3, color="violet", 
            #                  label="$\mathtt{FFT}(TT_{y, POL})$")
            ax3.semilogy(pol_freq, psd_pol_fft, linewidth=0.3, color="dodgerblue", 
                         label="Mean of $\mathtt{PSD}(TT_{x, POL})$ & $\mathtt{PSD}(TT_{y, POL})$")
            ax3.semilogy(pol_freq, psd_pol_fft_filt, linewidth=0.5, color="blue", 
                         label="SG(mean POL)")
            # Include CL for comparison
            ax3.semilogy(freq, psd_fft_filt, linewidth=0.5, color="grey", 
                         label="SG(mean CL)")
            # Plot vertical line where CL spectrum breaks/bends (equals POL spectrum)
            ax3.axvline(x=break_freqs[i], linestyle='--', linewidth=1.5, color='grey')
            ax3.set_title("Closed-loop (CL) vs Pseudo Open-loop (POL) tip-tilt power spectral densities (PSD)\nwith Savitzsky-Golay (SG) filter")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Log PSD")
            ax3.legend(ncol=2)
            ax3.sharex(ax2)
            ax3.set_xlim(0.0, 500.0) # Hz

            # Log-log version of CL vs POL PSD plot
            ax4.loglog(pol_freq, psd_pol_fft, linewidth=0.3, color="dodgerblue", 
                       label="Mean of $\mathtt{PSD}(TT_{x, POL})$ & $\mathtt{PSD}(TT_{y, POL})$")
            ax4.loglog(pol_freq, psd_pol_fft_filt, linewidth=0.5, color="blue", 
                       label="SG(mean POL)")
            # Include CL for comparison
            ax4.loglog(freq, psd_fft_filt, linewidth=0.5, color="grey", 
                       label="SG(mean CL)")
            # Plot vertical line where CL spectrum breaks/bends (equals POL spectrum)
            ax4.axvline(x=break_freqs[i], linestyle='--', linewidth=1.5, color='black')
            ax4.set_title("Closed-loop (CL) vs Pseudo Open-loop (POL) tip-tilt power spectral densities (PSD)\nwith Savitzsky-Golay (SG) filter")
            ax4.set_xlabel("Log Frequency (Hz)")
            ax4.set_ylabel("Log PSD")
            ax4.legend(ncol=2)

            # Welch plot
            ax5.semilogy(f, pxx_den_avg, linewidth=0.3, color="dodgerblue", 
                         label="Mean of $\mathtt{Welch}(TT_{x, CL})$ & $\mathtt{Welch}(TT_{y, CL})$")
            ax5.semilogy(f, pxx_den_sg, linewidth=0.3, color="blue", 
                         label="SG(Welch PSD CL)")
            ax5.semilogy(f, pxx_den_avg_pol, color='magenta',
                         label="Mean of $\mathtt{Welch}(TT_{x, POL})$ & $\mathtt{Welch}(TT_{y, POL})$")
            ax5.semilogy(f, pxx_den_sg_pol, color='red',
                         label="SG(Welch PSD POL)")
            # Plot vertical line where CL spectrum breaks/bends (equals POL spectrum)
            ax5.axvline(x=welch_break_freqs[i], linestyle='--', linewidth=1.5, color='black')
            ax5.set_title("CL vs POL TT PSDs via Welch's method")
            ax5.set_xlabel("Frequency (Hz)")
            ax5.set_ylabel("Log PSD")
            ax5.legend(ncol=2)

            # Separate plots for SHWFS telemetry timestream and TT APDCOUNTS timestream
            # Compare the two to infer whether cloud cover came in during exposure
            _, axis = plt.subplots(figsize=(17.5, 15.0), layout='constrained')
            # Have to choose specific subaperture (choose 50, reliably illuminated)
            plt.plot(data.a.timestamp[0]/(1e9), data.a.subapintensity[0][:,50]*0.508, '.-', 
                     color="blue", alpha=0.2)
            # plt.axhline(y=strapdqmn, color='red', linestyle='--', 
            #             linewidth=2, label=r"STRAPDQMN (quad mean adu)")
            # plt.text(ax1.get_xlim()[0], strapdqmn, s=f"{strapdqmn}", color='red')
            plt.title(f"SHWFS SUBAPINTENSITY [e-] on {dateobs} in {frame}")
            plt.xlabel(f"Time (sec) from beginning of {year}")
            axis.xaxis.set_major_formatter(FuncFormatter(sci_format))

            # _, axis = plt.subplots(figsize=(17.5, 15.0), layout='constrained')
            # plt.plot(data.b.timestamp[0]/(1e9), data.b.apdcounts, '.-', color="magenta", alpha=0.2)
            # plt.axhline(y=strapdqmn, color='red', linestyle='--', 
            #             linewidth=2, label=r"STRAPDQMN (quad mean adu)")
            # plt.text(ax1.get_xlim()[0], strapdqmn, s=f"{strapdqmn}", color='red')
            # plt.title(f"STRAP APDCOUNTS [adu] on {dateobs} in {frame}")
            # plt.xlabel(f"Time (sec) from beginning of {year}")
            # axis.xaxis.set_major_formatter(FuncFormatter(sci_format))
            plt.show()
    return avg_tt_centroid_x, avg_tt_centroid_y, avg_tt_laser_res, break_freqs

def Savitzky_Golay(data:list, window:int, order:int):
    """
    Function to reduce the noise of an input signal via
    smoothing through the Savitzky-Golay filter.

    from: https://plotly.com/python/smoothing/
    https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/

    Inputs:
    -------
    data   : 1d array, dtype=float
        Array with data of signal to be noise-reduced

    window : int
        Window size used for filtering (i.e., the number of coefficients). 
        If mode is ‘interp’ (default), window must be less than or equal to 
        the size of signal.

    order  : int
        Order of the fitted polynomial

    Outputs:
    --------
    signal.savgol_filter(data, window, order, mode='interp') : 1d array, dtype=float
        Array of filtered data

    By Brooke DiGia
    """
    return signal.savgol_filter(data, window, order, mode='interp')

def find_tt_break(cl:list, pol:list, freq:list, rtol:float=0.1):
    """
    Function to find the 'break' or 'bend' in a closed-loop tip-tilt
    power spectrum by comparing it to the pseudo-open-loop tip-tilt
    power spectrum

    np.isclose's equation for two floating points a and b:
    absolute(a - b) <= (atol + rtol * absolute(b))
    where atol is absolute tolerance and rtol is relative
    tolerance, as defined by np.isclose.
    (https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
    In this function we use rtol, default value of 0.1 is used as a
    percentage (i.e. 0.1*absolute(b))

    Inputs:
    --------
    cl   : 1d array, dtype=float
        Closed-loop TT power spectrum (FFT)

    pol  : 1d array, dtype=float
        Pseudo-open-loop TT power spectrum (FFT)

    freq : 1d array, dtype=float
        List of associated frequencies for input FFTs/power spectra
        (Hz)

    Outputs:
    --------
    freq[break_i] : float
        Frequency where break in CL spectra occurs

    By Brooke DiGia
    """
    compare = np.isclose(cl, pol, rtol=rtol)
    # First TRUE element in compare is first element
    # where CL and POL spectra are 'equal' (within 
    # tolerance). It is up to this frequency that we are
    # correcting
    try:
        break_i = list(compare).index(next(filter(lambda i: i == True, compare)))
    except StopIteration as itr_error:
        print(f"{itr_error} raised")
        # Check that list is indeed empty or filled with falses (no break
        # frequency found)
        if len(compare) == 0 or (not any(compare)):
            print("No break frequency found given relative tolerance {rtol}")
            return np.nan
    return freq[break_i]

def tt_noise_van_dam(f:list, cx:list):
    """
    Function to calculation the tip-tilt noise power spectrum
    from Marcos van Dam et al (Performance of the Keck
    Observatory adaptive-optics system, equation 24)

    Inputs:
    --------
    f  : 1D array, variable length
        Frequency array (Hz) for which to calculate |N_tt(f)|,
        the TT noise power spectrum
    
    cx : 1D array, len(f)
        Tip estimates in centroid units

    Outputs:
    ---------

    By Brooke DiGia
    """
    N_tt = np.sqrt( (2.0/240.0) * (12.68/1.2)**2.0 * np.var(cx))
    return N_tt

def DM_fitting_error(r0:u.m, alpha_f:float=0.46, d:u.mm=7*u.mm, 
                     wvl:u.nm=500.0*u.nm) -> u.nm:
    """
    Function to calculate the DM fitting error (to compare with MAOS
    error budget results), according to Marcos van Dam et al in
    Performance of the Keck Observatory adaptive-optics system.

    The fitting error is defined as the component of the wave-front
    that cannot be corrected by the SM. General form is given by
    equation 12 in the above paper:

    sigma_fit = sqrt(alpha_f) * (d / r0)**(5/6) * (lambda/(2*pi))
    where d is actuator spacing, lamba is the wavelength at which
    r0 is measured, and alpha_f is a constant dependent on the DM
    influence function (equation 3 in paper). alpha_f in the paper
    is calculated as 0.46.

    Inputs:
    -------
    r0       : float
        Fried parameter in meters
    
    alpha_f  : float, default=0.46
        Constant, set to value from aforementioned paper.
        Constant itself is calculated from the DM influence
        function provided in the paper, but that calculation
        is not replicated here

    d        : float, default=7 mm
        Actuator spacing in mm, from Keck Telescope and Instrument Guide

    wvl      : float, default=500 nm
        Wavelength at which r0 is calculated in nm

    Outputs:
    ---------
    sigma_DM : float
        The DM fitting error in nm

    By Brooke DiGia
    """
    return np.sqrt(alpha_f) * (d / r0)**(5/6) * (wvl / (2 * np.pi))

def telemetry_decimation(telem:object):
    """
    Function to check whether the telemetry header integration time keywords
    for STRAP and HO SHWFS match the telemetry data object timestamp data, to 
    evaluate whether the telemetry has been decimated or not

    Inputs:
    -------
    telem : object
       Data object from telemetry file, previously loaded outside this function

    Outputs:
    --------

    By Brooke DiGia 
    """
    frame = ''
    year = ''
    strap_int_time = 0.0
    shwfs_int_time = 0.0
    for item in telem.header:
        decoded:str = item.decode('ascii')
        if decoded.startswith('FILENAME'):
            frame = decoded.split(' ')[1][1:]
        elif decoded.startswith('DATE-OBS'):
            dateobs = decoded.split(' ')[1][1:]
            year = dateobs[:4]
        # STRAP integration time (ms)
        elif decoded.startswith('STINTTIM'):
            int_time = decoded.split(' ')[1]
            strap_int_time = float(int_time[1:])
        # SHWFS frame rate (Hz converted to ms)
        elif decoded.startswith('WSFRRT'):
            frame_rate = decoded.split(' ')[3]
            shwfs_int_time = (1.0/float(frame_rate[1:]))*1000.0 # ms

    # STRAP telemetry spacing
    strap_time_interval = np.mean(np.diff(telem.b.timestamp[0])) * 100.0 * (1e-9) * 1000.0 # ms
    shwfs_time_interval = np.mean(np.diff(telem.a.timestamp[0])) * 100.0 * (1e-9) * 1000.0 # ms
    print(f"STRAP header integration time = {strap_int_time} ms | STRAP telemetry interval = {strap_time_interval} ms")
    print(f"SHWFS header integration time = {shwfs_int_time} ms | SHWFS telemetry interval = {shwfs_time_interval} ms")

    # Return boolean evaluation for decimation - compare to hundredths place
    strap_decimated = np.isclose(strap_time_interval, strap_int_time)
    shwfs_decimated = np.isclose(shwfs_time_interval, shwfs_int_time)
    return strap_int_time, shwfs_int_time, strap_time_interval, shwfs_time_interval, strap_decimated, shwfs_decimated

# def TT_transfer_function(k_TT:float):
#     """
#     Function to calculate the TT loop transfer function,
#     given the TT loop gain.

#     Inputs:
#     -------
#     k_TT : float
#         Variable TT loop gain (from telemetry)

#     Outputs:
#     --------
#     H_TT : Transfer function

#     By Brooke DiGia
#     """
#     s = control.TransferFunction.s
#     H = (0.8 * k_TT * s) / (s - 1)
#     return H

def dark_current():
    """
    Function to calculate the dark current of a WFS, using the formula provided in KAON 387.
    """
    return

"""
The following *_on_sky() functons are copied and modified from the KAI repository, linked
above in function headers, for use on on-sky PSF images. This is to keep the
KAI pipeline unchanged in its own repository.
"""

def calc_strehl_on_sky(file_list, out_file, apersize=0.6, 
                       instrument=instruments.default_inst,
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
    if filt.lower() == "br_gamma":
        dl_img_file = cal_dir + "brgamma" + '.fits'
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
    # Calculate the peak flux ratio
    try:
        dl_peak_flux_ratio = calc_peak_flux_ratio_on_sky(dl_img, peak_coords_dl, 
                                                         radius, skysub)
        # For each image, get the strehl, FWHM, RMS WFE, MJD, etc. and write to an
        # output file.
        strehls = []
        fwhms = []
        rmswfes = []
        empfwhms = []
        for ii in range(len(file_list)):
            strehl, fwhm, rmswfe, emp_fwhm = calc_strehl_single_on_sky(file_list[ii], radius, 
                                                                       dl_peak_flux_ratio, 
                                                                       instrument=instrument, 
                                                                       skysub=skysub)
            strehls.append(strehl)
            fwhms.append(fwhm)
            rmswfes.append(rmswfe)
            empfwhms.append(emp_fwhm)
            mjd = fits.getval(file_list[ii], instrument.hdr_keys['mjd'])
            _, filename = os.path.split(file_list[ii])

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
    return strehls, fwhms, rmswfes, empfwhms

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
    # if 'XSTREHL' in hdr:
    #     coords = np.array([float(hdr['XSTREHL']),
    #                        float(hdr['YSTREHL'])])
    #     coords -= 1     # Coordinate were 1 based; but python is 0 based.
    
    # Calculate the FWHM using a 2D gaussian fit. We will just average the two.
    # To make this fit more robust, we will change our boxsize around, slowly
    # shrinking it until we get a reasonable value.

    # First estimate the DL FWHM in pixels. Use this to set the boxsize for
    # the FWHM estimation... note that this is NOT the aperture size specified
    # above which is only used for estimating the Strehl.
    dl_res_in_pix = 0.25 * wavelength / ( instrument.telescope_diam * scale )
    fwhm_min = dl_res_in_pix # * 0.9
    fwhm_max = 100.0
    fwhm = 0.0
    fwhm_boxsize = int(np.ceil((4 * dl_res_in_pix)))
    if fwhm_boxsize < 3:
        fwhm_boxsize = 3
    pos_delta_max = 2.0*fwhm_min
    box_scale = 1.0
    iters = 0

    # Steadily increase the boxsize until we get a reasonable FWHM estimate.
    while ((fwhm < fwhm_min) or (fwhm > fwhm_max)) and (iters < 50):
        box_scale += iters * 0.1
        iters += 1
        g2d = fit_gaussian2d(img, coords, fwhm_boxsize*box_scale,
                             fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                             pos_delta_max=pos_delta_max, plot=True)
        sigma = (g2d.x_stddev_0.value + g2d.y_stddev_0.value) / 2.0
        fwhm = stddev_to_fwhm(sigma)
        emp_fwhm = empirical_fwhm(img, scale)

        # print(f"FWHM on iteration {iters} = {fwhm:.2f} mas | Empirical FWHM on iteration {iters} = {emp_fwhm:.2f} mas")

        # print(img_file.split('/')[-1], iters, fwhm,
        #           g2d.x_mean_0.value, g2d.y_mean_0.value, fwhm_boxsize*box_scale)

        # Update the coordinates if they are reasonable. 
        if ((np.abs(g2d.x_mean_0.value - coords[0]) < fwhm_boxsize) and
            (np.abs(g2d.y_mean_0.value - coords[1]) < fwhm_boxsize)):
            coords = np.array([g2d.x_mean_0.value, g2d.y_mean_0.value])

    # Convert to milli-arcseconds
    fwhm *= scale * 1e3
    emp_fwhm *= 1.0e3

    # metrics = fit_gaussian2d_alternative(img, coords, scale)
    # fwhm = metrics['fwhm']*1e3 # mas
    # emp_fwhm = metrics['emp_fwhm']*1e3 # mas

    # Calculate the peak flux ratio
    peak_flux_ratio = calc_peak_flux_ratio_on_sky(img, coords, radius, skysub)

    # Normalize by the same from the DL image to get the Strehl.
    strehl = peak_flux_ratio / dl_peak_flux_ratio
    print('peak flux ratio = ', peak_flux_ratio, ' dl peak flux ratio = ', dl_peak_flux_ratio)

    # Convert the Strehl to a RMS WFE using the Marechal approximation.
    rms_wfe = np.sqrt( -1.0 * np.log(strehl)) * wavelength * 1.0e3 / (2. * math.pi)
    
    # Check final values and fail gracefully.
    if ((strehl < 0) or (strehl > 1) or
        (fwhm > 500) or (fwhm < (fwhm_min * scale * 1.0e3))):
        strehl = np.nan
        fwhm = np.nan
        rms_wfe = np.nan

    fmt_dat = '{img:<30s} {strehl:7.3f} {rms:7.1f} {fwhm:7.2f} {xpos:6.1f} {ypos:6.1f}\n'
    print(fmt_dat.format(img=img_file, strehl=strehl, rms=rms_wfe, fwhm=fwhm, xpos=coords[0], ypos=coords[1]))
    
    return strehl, fwhm, rms_wfe, emp_fwhm

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


def get_conditions_for_directories(directory_list, save_dimm_dir='./'):
    """
    Return a dictionary continiang one QTable for each directory in the
    input directory list. This table will contain things like MASS/DIMM, etc.
    (everything from get_atm_conditions).
    """
    from astropy import units as u
    from astropy.table import QTable
    from astropy.time import Time
    
    results_dir = {}
                   
    for ll in range(len(directory_list)):
        cfiles = glob.glob(directory_list[ll] + 'ci*.fits')

        N_files = len(cfiles)

        # Load up the Strehl, FWHM file produced by KAI
        strehl_tab = Table.read(directory_list[ll] + 'strehl_source.txt', format='ascii')

        for ff in range(len(cfiles)):
            results = estimate_on_sky_conditions(cfiles[ff], save_dimm_dir)

            if ff == 0:
                file_names = np.zeros(N_files, dtype='S30')
                time_mjd = np.zeros(N_files, dtype='float')
                strehl = np.zeros(N_files, dtype='float')
                fwhm = np.zeros(N_files, dtype='float')
                rmswfe = np.zeros(N_files, dtype='float')
                r0_start = np.zeros(N_files, dtype='float') * u.m
                start_turb = np.zeros((N_files, len(results[1])), dtype='float')
                wind_spd_profile = np.zeros((N_files, len(results[2])), dtype='float')
                wind_dir_profile = np.zeros((N_files, len(results[2])), dtype='float')
                closest_dimm_start = np.zeros(N_files, dtype='float')
                mass_profile_start = np.zeros(N_files, dtype='float')
                time_of_dimm = np.zeros(N_files, dtype='10S')
                time_of_mass = np.zeros(N_files, dtype='10S')
                tau_0 = np.zeros(N_files, dtype='float') * u.s
                theta_0 = np.zeros(N_files, dtype='float') * u.arcsec
                sigma_DM = np.zeros(N_files, dtype='float') * u.nm

                tab = QTable([file_names, time_mjd, strehl, fwhm, rmswfe,
                              r0_start, start_turb, wind_spd_profile, wind_dir_profile,
                             closest_dimm_start, mass_profile_start, time_of_dimm, time_of_mass,
                             tau_0, theta_0, sigma_DM],
                            names = ['file', 'time_mjd', 'strehl', 'fwhm', 'rmswfe',
                                     'r0', 'turb', 'wind_spd_profile', 'wind_dir_profile',
                                     'dimm', 'mass', 'time_of_dimm', 'time_of_mass', 
                                     'tau_0', 'theta_0', 'sigma_DM'])
            


            tab['file'][ff] = cfiles[ff].split('/')[-1]
            tab['r0'][ff] = results[0]
            tab['turb'][ff] = results[1]
            tab['wind_spd_profile'][ff] = results[2]
            tab['wind_dir_profile'][ff] = results[3]
            tab['dimm'][ff] = results[4]
            tab['mass'][ff] = results[5]
            tab['time_of_dimm'][ff] = results[6]
            tab['time_of_mass'][ff] = results[7]
            tab['tau_0'][ff] = results[8]
            tab['theta_0'][ff] = results[9]
            tab['sigma_DM'][ff] = results[10]

            # Get exposure time in hours
            hdr = fits.getheader(cfiles[ff])
            tab['time_mjd'][ff] = hdr['MJD-OBS']

            # Get the Strehl, FWHM, RMSWFE
            idx = np.where(strehl_tab['Filename'] == tab['file'][ff])[0]

            if len(idx) > 0:
                tab['strehl'][ff] = strehl_tab['Strehl'][idx[0]]
                tab['rmswfe'][ff] = strehl_tab['RMSwfe'][idx[0]]
                tab['fwhm'][ff] = strehl_tab['FWHM'][idx[0]]

        results_dir[directory_list[ll]] = tab

    return results_dir
