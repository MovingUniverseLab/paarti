import numpy as np
import pylab as plt
import scipy
import math
import poppy
from paarti.utils import maos_utils
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import os

# import readbin from MAOS
import readbin

def make_keck_vib_psd(jitter_tot=1.0):
    """
    Prepare a Keck wind-shake and vibration PSD modeled after an
    exiting TMT one. The user can scale this PSD with the 
    optional jitter input [mas], which will go into the PSD
    filename.

    Inputs:
    ----------
    jitter_tot            : float, default = 1.0
        New total jitter [mas] for PSD to be generated
        
    Outputs:
    ----------
    psd_outfoot + '.fits' : string
	Filename of generated PSD

    Modified by Brooke DiGia
    """
    maos_config_dir = os.environ['MAOS_CONFIG']
    maos_psd_dir = maos_config_dir + '/maos/bin/'    
    psd_input_file = maos_psd_dir + 'PSD_TMT_ws20mas_vib15mas_rad2.bin'
    
    # Location of vibration bump (29 Hz)
    vib_freq = 29

    # 1D RMS from jitter disturbances per Figure 4.5 of KAON1303
    # vib_jitter_amp_tot = (45 + 48)  / 2.0  # mas

    # Override since this looks like it is getting partially corrected in MAOS 
    vib_jitter_amp_tot = ( 20.0**2.0 + 15.0**2.0 + 7.0**2.0 )**0.5 # mas
    
    # TMT PSD already contains windshake and a broad vibration spectrum. 
    # No peaks though. Subtract out the jitter already in the TMT PSD to get 
    # the amount we need to add in the 29 Hz bump. 
    vib_jitter_amp = np.sqrt( vib_jitter_amp_tot**2.0 - 20.0**2.0 - 15.0**2.0 )
    print(f'Adding {vib_jitter_amp:.1f} mas bump at {vib_freq} Hz...')

    # Convert to arcseconds
    vib_jitter_amp /= 1e3

    # Add new vibrations
    freq, psd = maos_utils.psd_add_vibrations(psd_input_file, vib_freq, vib_jitter_amp)

    # Calculate area of this PSD
    A = np.square(maos_utils.psd_integrate_sqrt(freq, psd).to("mas"))

    # Scale PSD (if user did not specify this argument, the default value is 1.0,
    # which is equivalent to no scaling of the original PSD)
    psd *= ( (jitter_tot*u.mas)**2.0 ) / A

    # Plot PSD
    plt.close(1)
    plt.figure(1)
    plt.loglog(freq, psd.to( u.mas**2 / u.Hz ))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Jitter PSD (mas^2/Hz)')
    plt.ylim(1e-3, 1e3)
    print(f'Total jitter = {jitter_tot} mas')

    psd_outroot = f'{maos_psd_dir}PSD_Keck_ws{jitter_tot}mas_vib{vib_jitter_amp_tot:.0f}mas_rad2'
    plt.savefig(psd_outroot + '.png')
    
    # Create a 2D array with freq and power as expected by MAOS.
    freq_psd = np.array([freq, psd])
    fits.writeto(psd_outroot + '.fits', freq_psd, overwrite=True)

    # Return filename of generated PSD for grid search purposes
    return psd_outroot + '.fits'

def make_keck_ncpa(rms_wfe, seg_piston_file='Keck_Segment_Pistons_2023_04_29.csv'):
    """
    Prepare a Keck NCPA map capturing primary mirror phasing errors that
    can't be calibrated out.

    Inputs
    ------
    rms_wfe : float
        Desired RMS WFE in nm. The input segment pistons will just be rescaled
        to yield the desired RMS WFE.

    Optional Inputs
    ---------------
    seg_piston_file : str
        Name of the CSV file containing the piston for each primary mirror segment.
    """
    inner_ring = 1
    outer_ring = 3
    in_edge_length = 89.88453 * u.cm
    in_gap_size = 7 * u.mm
    sec_rad = 2.48 * u.m / 2   # secondary alone is 1.4 m diam, but support structure adds another 1m
    spider_num = 6
    spider_width = 0.0254 * u.m 

    # Define primary. Rotations are needed to get into the Keck segment ID convention later.
    # Note, we do something slightly different than for the true Keck pupil as we ignore
    # spiders, ignore secondary obscuration, etc. 
    prim_as_dm = poppy.dms.HexSegmentedDeformableMirror(rings=outer_ring,
                                                        side=in_edge_length, gap=in_gap_size,
                                                        rotation=-30)
    prim_as_dm.pupil_diam *= u.m  # just fixing poppy bug.

    # Read in segment piston map from Zernike WFS measurements (K2).
    # File should contain three columns.
    #     Segment index, Mean Piston in nm, StdDev Piston in nm
    pist = Table.read(seg_piston_file)
    
    # Rescale RMS WFE to the desired value.
    rms_wfe_orig = np.nanstd(pist['Mean']) # nm
    print(f'Original Surface RMS WFE = {rms_wfe_orig:.1f} nm')

    pist['Mean'] = pist['Mean'] * rms_wfe / rms_wfe_orig
    print(f'New Surface RMS WFE (in) = {rms_wfe:.1f} nm')
    print(f'New Surface RMS WFE (emp)= {np.nanstd(pist["Mean"]):.1f} nm')
    
    # Loop through Keck segment IDs and add WFE (in OPD).
    # Note, Poppy's ID scheme is slightly different.
    for ii in range(len(pist)):
        prim_as_dm.set_actuator(pist['SegID'][ii], pist['Mean'][ii]*u.nm, 0, 0)

    # Debugging to figure out actuator numbers
    # prim_as_dm.set_actuator(1, pist['Mean'][ii]*100*u.nm, 0, 0)
    # prim_as_dm.set_actuator(2, pist['Mean'][ii]*50*u.nm, 0, 0)
    # prim_as_dm.set_actuator(7, pist['Mean'][ii]*25*u.nm, 0, 0)

    # Now add secondary with spiders.
    # Rotation is needed to get into Keck segment ID convention.
    sec = poppy.SecondaryObscuration(secondary_radius=sec_rad,
                                     n_supports=spider_num, support_width=spider_width,
                                     rotation=-30)

    # combine into one optic
    atlast = poppy.CompoundAnalyticOptic(opticslist=[prim_as_dm, sec], name='Keck')

    # Use 11 m sized rectangular array of the pupil.
    diam = 11 * u.m
    temp_wavefront = poppy.Wavefront(wavelength=2*u.micron, npix=1024, diam=diam)
    
    print(f'Pupil image size is {diam}')

    # Final OPD map.
    opd = atlast.get_opd(temp_wavefront).copy()

    # Plot the OPD map of NCPA.
    plt.figure(1, figsize=(9,8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95)
    plt.clf()
    minmax = np.max(np.abs(opd))
    plt.imshow(opd.T * 1e9, cmap='seismic', vmin=-minmax*1e9, vmax=minmax*1e9)
    plt.axis('equal')
    plt.colorbar(label='OPD (nm)')
    for seg_id in prim_as_dm._seg_indices:
        # Invert coordinates since we transposed the array.
        seg = prim_as_dm._seg_indices[seg_id]
        y = seg[1].mean()
        x = seg[0].mean()
        plt.text(x, y, f'{seg_id}', ha='center', va='center', fontsize=10)

    # Output directory for images and NCPA map.
    maos_config_dir = os.environ['MAOS_CONFIG']
    maos_ncpa_dir = maos_config_dir + '/maos/bin/'
    
    outroot = f'{maos_ncpa_dir}/Keck_ncpa_rmswfe{rms_wfe:.0f}nm'

    plt.savefig(outroot + '.png')

    hdr = fits.Header()
    diam_m = diam.to('m').value
    hdr['dx'] = diam_m / opd.shape[0]
    hdr['dy'] = diam_m / opd.shape[1]
    hdr['ox'] = -diam_m / 2.0
    hdr['oy'] = -diam_m / 2.0
    hdr['h'] = 0

    fits.writeto(outroot + '.fits', opd, hdr, overwrite=True)

    return

def make_keck_pupil():
    """
    Generates a fits file corresponding to the Keck pupil. 
    """

    #Grab the poppy keck pupil
    atlast = make_keck_poppy_pupil()

    # Use 11 m sized rectangular array of the pupil.
    diam = 11 * u.m
    temp_wavefront = poppy.Wavefront(wavelength=2*u.micron, npix=1024, diam=diam)
    
    print(f'Pupil image size is {diam}')
    ampl = atlast.get_transmission(temp_wavefront)
    transmittance = ampl**2

    # Transpose the array to get it into the angle=0 convention
    # of Keck. (see the NCPA maps that must match).
    transmittance = transmittance.T

    plt.figure(2)
    plt.clf()
    plt.imshow(transmittance)

    hdr = fits.Header()
    diam_m = diam.to('m').value
    hdr['dx'] = diam_m / transmittance.shape[0]
    hdr['dy'] = diam_m / transmittance.shape[1]
    hdr['ox'] = -diam_m / 2.0
    hdr['oy'] = -diam_m / 2.0
    hdr['h'] = 0

    fits.writeto('KECK_gaps_spiders.fits', transmittance, hdr, overwrite=True)

    return

def make_keck_poppy_pupil():
    '''
    Makes a poppy pupil object matching the Keck Pupil. 

    From Arroyo (Matthew Britton):
    Construct an aperture like at Keck,
    with rings of hexagonal apertures.
    The central hexagon is labelled as ring
    zero.  So for the Keck primary,
    inner_ring = 1 and outer_ring = 3

    According to Mitch, the actual glass 
    hexagon has a 90 cm edge.  But the outer
    1 mm is beveled, so is optically opaque.
    Then there's a 5 mm gap between hexes.
    So its like having a 7 mm gap, with a hexagon
    that has an edge length of 
    90 cm - 2*(.1 cm) /sqrt(3) = 89.88453 cm

    Note - edge length and gap size must be 
    positive.  Both are in meters.

    See https://arxiv.org/pdf/2109.00612.pdf for more
    recent details and measurements of the pupil.
    
    Returns a poppy CompoundAnalyticOptic
    '''

    # inner_ring = 1 #Not used
    outer_ring = 3
    in_edge_length = 89.88453 * u.cm
    in_gap_size = 7 * u.mm
    sec_rad = 2.48 * u.m / 2   # secondary alone is 1.4 m diam, but support structure adds another 1m
    spider_num = 6
    spider_width = 0.0254 * u.m 

    # 3 rings of segments yields
    ap = poppy.MultiHexagonAperture(rings=outer_ring, side=in_edge_length, gap=in_gap_size)
    # secondary with spiders
    sec = poppy.SecondaryObscuration(secondary_radius=sec_rad,
                                     n_supports=spider_num, support_width=spider_width)

    ap.pupil_diam *= u.m
    
    # combine into one optic
    atlast = poppy.CompoundAnalyticOptic(opticslist=[ap, sec], name='Keck')

    return atlast
    
def generate_keck_psf(wavelength,pixelscale,fov=2.0):
    '''
    Generate a perfect monochromatic Keck PSF using the poppy 

    Inputs: 
    wavelength - in meters
    pixelscale - in arcsecond
    fov - field of view in arcseconds

    '''

    #TODO Check for astropy units. Consider forcing the inputs to have units

    osys = poppy.OpticalSystem()
    osys.add_pupil(make_keck_poppy_pupil())  

    osys.add_detector(pixelscale=pixelscale*u.arcsec/u.pixel, fov_arcsec=fov*u.arcsec)

    psf = osys.calc_psf(wavelength)   

    return psf

