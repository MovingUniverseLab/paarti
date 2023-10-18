import numpy as np
import math
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from paarti.psf_metrics import metrics
import glob
import readbin
from scipy import stats
import scipy

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
    '''
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

    Notes
    -----
    
    By Matthew Freeman and Paolo Turri

    Equations 65-68 from section 3B of
    Clare, R. et al (2006). Adaptive optics sky coverage modelling for
    extremely large telescopes. Applied Optics 45, 35 (8964-8978)

    by Brooke:
    Inputs: m is magnitude, wfs is string name for wave-front sensor,
    and optional input wfs_int_time refers to the integration time of
    said wave-front sensor.
    '''
    # new camera setups by Brooke:
    # LGSWFS-OCAM2K refers to 'KAPA' and 'KAPA + HODM' setups in simulation nomenclature
    # LGS-HODM-HOWFS refers to 'KAPA + New HODM, New HOWFS'
    # 'KAPA + New HODM, New HOWFS, New LGS' requires adjusted magnitude input
    # but not a separate elif branch, as all parameters are the same as
    # LGS-HODM-HOWFS
    # NewLGS refers to 'KAPA + New LGS'
 
    wfs_list = ['LBWFS', 'LGSWFS', 'LGSWFS-OCAM2K', 'LGS-HODM-HOWFS', 'NewLGS-HODM-HOWFS', 'NewLGS', 'TRICK-H', 'TRICK-K', 'STRAP']

    if wfs not in wfs_list:
        raise RuntimeError("keck_nea_photons: Invalid WFS.")

    # telescope diameter (m)
    # D = 11.8125
    D = 10.949
    # Secondary obscuration diameter (m)
    Ds = 1.8

    # Integration time on the WFS. Should be
    # 1/800 sec for LGS
    # 30 sec for LBWFS -- adjust with guide star brightness.
    # 1/800 sec for TT approximately
    time = wfs_int_time
    
    # r_0 = 0.178 # Fried parameter.
    # Median 17.8cm at 0.5 microns, from KAON303
    # Median 22.0 cm at 0.5 micron from Raven paper (Yoshito 2017)
    # Carlos' script uses 0.16*cos(parm.atm.zenithAngle )^(3/5); % coherence length in meters at 0.5microns
    
    # wavelength = guide star imaging wavelength
    # ps = pixel scale (arcsec).  
    # sigma_e = rms detector read noise per pixel
    # theta_beta = spot size on detector (radians)
    # pix_per_ap = pixels per subaperture, for noise calculation

    if wfs == 'LBWFS':
        band = "R"
        wavelength = 0.641e-6

        # side length of square subaperture (m).
        side = 0.5625 

        # 1.5 for WFS and low bandwithth WFS from Blake's config
        ps = 1.5
        
        # from Carlos' config
        sigma_e = 7.96
        
        # from KAON 1303 Table 16, sub-aperture spot size
        theta_beta = 0.49 *(math.pi/180)/(60*60)
        
        # from KAON 1303 Table 8
        throughput = 0.03
        
        # quadcell
        pix_per_ap = 4
        
    elif wfs == 'LGSWFS':
        band = "R"    # not actually at V.
        wavelength = 0.589e-6
        
        # side length of square subaperture (m). Should be equal to powfs.dsa and dm.dx in
        # respective config files.
        side = 0.563
        
        # From Carlos' config file
        ps = 3.0
        sigma_e = 3.0
        
        # From KAON 1303  Table 20
        theta_beta = 1.5 *(math.pi/180)/(60*60)
        
        # KAON 1303 Table 8 states 0.36, but Np=1000 is already
        # measured on the detector.
        # Modified to account for QE=0.88 on the WFS detector at R-band.
        # from error budget spreadsheet.
        throughput = 0.36 * 0.88
        
        # quadcell
        pix_per_ap = 4
        
    elif wfs == 'LGSWFS-OCAM2K':
        band = "R"    # not actually at V.
        wavelength = 0.589e-6
        
        # side length of square subaperture (m).
        side = 0.563
        
        # From Carlos' config file
        ps = 3.0
        sigma_e = 0.5
        
        # From KAON 1303  Table 20
        theta_beta = 1.5 *(math.pi/180)/(60*60)
        
        # KAON 1303 Table 8 states 0.36, but Np=1000 is already
        # measured on the detector.    
        # Modified to account for QE=0.88 on the WFS detector at R-band.
        # from error budget spreadsheet.
        throughput = 0.36 * 0.88
        
        # quadcell
        pix_per_ap = 4

    elif wfs == 'LGS-HODM-HOWFS':
        band = "R"
        wavelength = 0.589e-6
        side = 0.17
        ps = 3.0
        sigma_e = 0.1
        theta_beta = 1.5 * (math.pi/180)/(60*60)
        throughput = 0.36 * 0.88
        pix_per_ap = 4
  
    elif wfs == 'TRICK-H':
        band = "H"
        wavelength = 1.63e-6

        # side length of square subaperture (m)
        # turn into square aperture of same area as primary.
        side = math.sqrt( math.pi * ((D  / 2.0)**2 - (Ds / 2.)**2) )

        # From Carlos' config file
        ps = 0.06
        sigma_e = 4
        # Modified to get SNR=5 at H=15
        sigma_e = 11.188

        # Using OSIRIS FWHM from KAON 1303 Table 13 (as suggested by Peter)
        theta_beta = 0.055 *(math.pi/180)/(60*60)

        # from KAON 1303 Table 8
        throughput = 0.56
        # Modify to add 4 lenses and a filter inside TRICK.
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95
        
        # ROI reduces from 16x16 to 2x2 as residual is reduced.
        pix_per_ap = 4
        
    elif wfs == 'TRICK-K':
        band = "K"
        wavelength = 2.19e-6

        # side length of square subaperture (m)        
        # side = D  
        side = math.sqrt( math.pi * ((D  / 2.0)**2 - (Ds / 2.)**2) )
        
        # From Carlos' config file
        ps = 0.04
        sigma_e = 4
        # Modified to get SNR=5 at H=15
        sigma_e = 11.188
        
        # Scaling the K band 0.055 by 2.19/1.63 (wavelength ratio)        
        theta_beta = 0.074 *(math.pi/180)/(60*60)

        # from KAON 1303 Table 8
        throughput = 0.62
        # Modify to add 4 lenses and a filter inside TRICK.
        # TODO: Need to put in detector QE
        throughput *= 0.96**4 * 0.95

        # ROI decreases from 16x16 to 2x2 as residual reduces
        pix_per_ap = 4

    elif wfs == 'STRAP':
        band = "R"
        wavelength = 0.641e-6

        # side length of square subaperture (m)        
        # side = D  
        side = math.sqrt( math.pi * ((D  / 2.0)**2 - (Ds / 2.)**2) ) / 2.0
        
        # From KAON 1322, just above equation 19.
        ps = 1.3
        # Made up... everything seems limited by photon noise or background noise.
        sigma_e = 0.1

        ## There appears to be inconsistencies in that KAON 1322 Section 7.6
        # which quotes 3000 photons/aperture/frame (not sure what brightness star
        # this would be for). Maybe GC R=15?
        
        # from KAON 1303 Table 16
        theta_beta = 0.49 *(math.pi/180)/(60*60)

        # from KAON 1303 Table 7
        # Modified to account for QE=0.50 on the WFS detector at R-band
        # from error budget spreadsheet.
        throughput = 0.32 #* 0.50

        # ROI
        pix_per_ap = 4
            
    else:
        print("Other wfs:", wfs)


    SNR, sigma_theta, Np, Nb = keck_nea_photons_any_config(wfs,
                                                           side,
                                                           throughput,
                                                           ps,
                                                           theta_beta,
                                                           band,
                                                           sigma_e,
                                                           pix_per_ap,
                                                           time,
                                                           m)

    return SNR, sigma_theta, Np, Nb

def keck_nea_photons_any_config(wfs, side, throughput, ps, theta_beta, band, sigma_e, pix_per_ap, time, m):
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
    
    # number of photons and background photons (instead of eqs 69 and 70)
    Np, Nb = n_photons(side, time, m, band, ps, throughput)

    # Fix LGS background
    if 'LGSWFS' in wfs:
        Nb = 6   # Was

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
    
    SNR = Np /np.sqrt(Np + pix_per_ap*Nb + pix_per_ap*sigma_e**2)

    # noise equivalent angle in milliarcseconds (eq 65)
    sigma_theta = theta_beta/SNR  * (180/math.pi) *60*60*1000

    ### Print Outputs ###
    ####
    # Show results
    ####
    print('Outputs:')
    print(f"  N_photons from star (powfs.siglev for config files):                   {Np:.3f}")
    print(f"  N_photons per pixel from background:   {Nb:.3f}")
    print(f"  SNR:                                   {SNR:.3f}")
    print(f"  NEA (powfs.nearecon for config files):                                   {sigma_theta:.3f} mas")
    
    return SNR, sigma_theta, Np, Nb
    

def n_photons(side, time, m, band, ps, throughput):
    """Calculate the number of photons from a star and
       background incident on a square area in a given time interval.

    Author: Paolo Turri
        
    Bibliography:
    [1] Bessel et al. (1998)
    [2] Mann & von Braun (2015)
    [3] https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_%28Sec_2%29.html
    """

    # Parameters
    # side = 0.5625  # Side of the square area (m)
    # time = 1/800  # Time interval (s)
    # m = 7.3 # Apparent magnitude (Vega system)
    # band = "K"  # Band name ("U", "B", "V", "R", "I", "J", "H", "K")
    # ps = 0.04  # Pixel scale (arcsec px^-1)

    # Fixed parameters
    c = 2.99792458e8  # Speed of light (m s^-1)
    h = 6.6260755e-27  # Plank constant (erg s)
    bands = {'name': ["U", "B", "V", "R", "I", "J", "H", "K"],
             'lambd': [0.366, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19],
             'delta_lambd': [0.0665, 0.1037, 0.0909, 0.1479, 0.1042, 0.3268, 0.2607,
                             0.5569],
             'phi_erg': [417.5, 632, 363.1, 217.7, 112.6, 31.47, 11.38, 3.961],
             'bkg_m': [21.6, 22.3, 21.1, 20.3, 19.2, 14.8, 13.4, 12.6]}
    # Bands' names, effective wavelengths (um), equivalent widths (um), fluxes
    # (10^-11 erg s^-1 cm^-2 A^-1) and background in (magnitudes arcsec^-2) [1, 2, 3].

    ####
    # Get band's data
    ####
    band_idx = np.where(np.array(bands['name']) == band)[0][0]
    
    # Band effective wavelength (um)    
    lambd = float(bands['lambd'][band_idx])
    # Band equivalent width (um)
    delta_lamb = float(bands['delta_lambd'][band_idx])
    # Flux (erg s^-1 cm^-2 A^-1)    
    phi_erg = float(bands['phi_erg'][band_idx])
    # Background magnitude (arcsec^-2)    
    bkg_m = float(bands['bkg_m'][band_idx])

    ####
    # Calculations of the star
    ####
    # Band frequency (s^-1)
    f = c / (lambd * 1e-6)
    # Numeric flux (s^-1 cm^-2 A^-1)
    phi_n = phi_erg * 1e-11 / (h * f)
    # Zeropoint (m = 0) number of photons on detector
    n_ph_0 = phi_n * ((side * 1e2) ** 2) * time * delta_lamb * 1e4 * throughput  
    # Number of star photons
    n_ph_star = n_ph_0 * (10 ** (-0.4 * m)) 

    ####
    # Calculations of the background
    ####
    # Number of background photons (px^-1)
    n_ph_bkg = n_ph_0 * (10 ** (-0.4 * bkg_m)) * (ps ** 2)  
    
    return n_ph_star, n_ph_bkg

def keck_ttmag_to_itime(ttmag, wfs='strap'):
    """
    Calculate the expected integration time for STRAP given
    a tip-tilt star magnitude in the R-band.

    Inputs
    ------
    ttmag : float
        Tip-tilt star brightness in apparent R-band magnitudes in
        the Vega system.

    Ouptputs
    --------
    itime : float
        The integration time used for STRAP in seconds.
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
    results_file = f'{directory}Res_{seed}.bin'
    results = readbin.readbin(results_file)
    print("Looking in directory:", directory)

    # Open-loop WFE: Piston removed, TT only, Piston+TT removed
    open_mean_nm = results[0].mean(axis=0)**0.5 * 1e9   # in nm

    # Closed-loop WFE: Piston removed, TT only, Piston+TT removed
    clos_mean_nm = results[2].mean(axis=0)**0.5 * 1e9   # in nm

    print('---------------------')
    print('WaveFront Error (nm): [note, piston removed from all]')
    print('---------------------')
    print(f'{"      ":<7s}  {"Total":>11s}  {"High_Order":>11s}  {"TT":>11s}')
    print(f'{"Open  ":<7s}  {open_mean_nm[0]:11.1f}  {open_mean_nm[2]:11.1f}  {open_mean_nm[1]:11.1f}')
    print(f'{"Closed":<7s}  {clos_mean_nm[0]:11.1f}  {clos_mean_nm[2]:11.1f}  {clos_mean_nm[1]:11.1f}')

    return open_mean_nm, clos_mean_nm
    
def print_psf_metrics_x0y0(directory='./', oversamp=3, seed=10, cut_radius=20):
    """
    Print some PSF metrics for a central PSF computed by MAOS
    at an arbitrary number of wavelengths.
    """
    print("Looking in directory:", directory)  
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
        wavelengths[pp] = hdr["WVL"] * 1e6
        strehl_values[pp] = mets["strehl"]
        fwhm_gaus_values[pp] = mets["emp_fwhm"] * 1e3
        fwhm_emp_values[pp] = mets["fwhm"] * 1e3
        r_ee80_values[pp] = mets["ee80"] * 1e3

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
    at an arbitrary number of wavelengths.
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
    Read a MAOS PSD file and return the frequency and PSD arrays
    with units. Note, there are two types... a "jitter" file usually
    input for windshake and vibrations, which is in units of
    radian^2 / Hz. The second is a the residual WFE PSD output by
    MAOS in units of m^2 / Hz.
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
    

def psd_add_vibrations(psd_input_file, vib_freq, vib_jitter_amp):
    """
    Take an input temporal power-spectral density (PSD) function in rad^2/Hz
    (as expected for MAOS) and modify it to add a vibration peak with
    a log-normal distribution peaked at the input vibration frequency
    and with an integrated jitter amplitude equal to the specified value
    in arcseconds.

    Inputs
    ------
    psd_input_file : string
        The name of the input PSD file for wind-shake and vibrations in
        MAOS format. This is a 2 column binary array with the first column
        containing frequency in Hz and the second column containing the
        PSD in radian^2 / Hz. The file should be readable by the MAOS
        readbin utility or it should be a FITS file.

    vib_freq : float
        Peak vibration frequency in Hz.

    vib_jitter_amp : float
        Integrated jitter in arcsec over the whole vibration peak.
    """
    freq, psd = read_maos_psd(psd_input_file)
    
    dfreq = np.diff(freq)

    # Create a vibration peak
    vib_model = stats.lognorm(0.1, scale=vib_freq, loc=1)
    vib_model_psd = vib_model.pdf(freq) * u.radian**2 / u.Hz

    # Normalize the vibration peak to have a total jitter as specified.
    norm = np.sqrt(np.sum(vib_model_psd[1:] * dfreq)).to('arcsec')
    vib_model_psd *= (vib_jitter_amp * u.arcsec / norm)**2

    psd += vib_model_psd

    return freq, psd

    
def psd_integrate_sqrt(freq, psd):
    """
    Integrate and sqrt a PSD to give the total WFE or jitter. 
    """
    dfreq = np.diff(freq)

    total_variance = scipy.integrate.trapezoid(psd, freq)

    total_rms = total_variance**0.5

    return total_rms
    
