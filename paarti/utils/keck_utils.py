import numpy as np
import pylab as plt
import scipy
import math
import poppy
from paarti.utils import maos_utils
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

# import readbin from MAOS
import readbin

def prep_keck_vib_psd():
    """
    Prepare a Keck wind-shake and vibration PSD modeled after an
    exiting TMT one. 
    """
    psd_input_file = 'PSD_TMT_ws20mas_vib15mas_rad2.bin'
    vib_freq = 29 # Hz

    # 1D RMS from jitter disturbances per Figure 4.5 of KAON1303
    #vib_jitter_amp_tot = (45 + 48)  / 2.0  # mas

    # Override since this doesn't look like it is getting corrected in MAOS. 
    vib_jitter_amp_tot = (20**2 + 15**2 + 7**2)**0.5 # mas
    
    # TMT PSD already contains windshake and a broad vibration spectrum. No peaks though. 
    # Subtract out the jitter already in the TMT PSD to get the
    # amount we need to add in the 29 Hz bump. 
    vib_jitter_amp = np.sqrt( vib_jitter_amp_tot**2 - 20**2 - 15**2 )
    print(f'Adding {vib_jitter_amp:.1f} mas bump at {vib_freq}')

    vib_jitter_amp /= 1e3 # convert to arcsec

    freq, psd = maos_utils.psd_add_vibrations(psd_input_file, vib_freq, vib_jitter_amp)

    plt.figure()
    plt.loglog(freq, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Jitter PSD (mas^2/Hz)')
    print(f'Total jitter {maos_utils.psd_integrate_sqrt(freq, psd).to("mas")}')

    # Create a 2D array with freq and power as expected by MAOS.
    freq_psd = np.array([freq, psd])
    fits.writeto(f'PSD_Keck_ws20mas_vib{vib_jitter_amp_tot:.0f}mas_rad2.fits', freq_psd, overwrite=True)

    return

def prep_keck_ncpa(rms_wfe, seg_piston_file='Keck_Segment_Pistons_2023_04_29.csv'):
    """
    Prepare a Keck NCPA map capturing primary mirror phasing errors that
    can't be calibrated out.

    Inputs
    ------
    rms_wfe : float
        Desired RMS WFE in nm. The input segment pistons will just be rescaled
        to yield the desired RMS WFE.
    """
    inner_ring = 1
    outer_ring = 3
    in_edge_length = 89.88453 * u.cm
    in_gap_size = 7 * u.mm
    sec_rad = 2.48 * u.m / 2   # secondary alone is 1.4 m diam, but support structure adds another 1m
    spider_num = 6
    spider_width = 0.0254 * u.m 

    # Define primary. Rotations are needed to get into the Keck segment ID convention later.
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
    print(f'New Surface RMS WFE      = {rms_wfe:.1f} nm')
    
    # Loop through Keck segment IDs and add WFE (in OPD).
    # Note, Poppy's ID scheme is slightly different.
    for ii in range(len(pist)):
        prim_as_dm.set_actuator(pist['SegID'][ii], pist['Mean'][ii]*u.nm, 0, 0)

    # Debugging to figure out actuator numbers
    # prim_as_dm.set_actuator(1, pist['Mean'][ii]*100*u.nm, 0, 0)
    # prim_as_dm.set_actuator(2, pist['Mean'][ii]*50*u.nm, 0, 0)
    # prim_as_dm.set_actuator(7, pist['Mean'][ii]*25*u.nm, 0, 0)

    # Secondary with spiders. Rotation is needed to get into Keck segment ID convention.
    sec = poppy.SecondaryObscuration(secondary_radius=sec_rad,
                                     n_supports=spider_num, support_width=spider_width,
                                     rotation=-30)

    
    # combine into one optic
    atlast = poppy.CompoundAnalyticOptic(opticslist=[prim_as_dm, sec], name='Keck')
    # atlast.display(npix=1024, colorbar_orientation='vertical')

    # Use 11 m sized rectangular array of the pupil.
    diam = 11 * u.m
    temp_wavefront = poppy.Wavefront(wavelength=2*u.micron, npix=1024, diam=diam)
    
    print(f'Pupil image size is {diam}')
    # ampl = atlast.get_transmsision(temp_wavefront)

    opd = atlast.get_opd(temp_wavefront).copy()
    # opd[(ampl == 0)] = np.nan    

    # plt.figure(2)
    # plt.clf()
    # atlast.display(what='both', npix=1024)

    plt.figure(1, figsize=(9,8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95)
    plt.clf()
    minmax = np.max(np.abs(opd))
    plt.imshow(opd.T, cmap='seismic', vmin=-minmax, vmax=minmax)
    plt.axis('equal')
    plt.colorbar(label='OPD (m)')
    for seg_id in prim_as_dm._seg_indices:
        # Invert coordinates since we transposed the array.
        seg = prim_as_dm._seg_indices[seg_id]
        y = seg[1].mean()
        x = seg[0].mean()
        plt.text(x, y, f'{seg_id}', ha='center', va='center', fontsize=10)

    hdr = fits.Header()
    diam_m = diam.to('m').value
    hdr['dx'] = diam_m / opd.shape[0]
    hdr['dy'] = diam_m / opd.shape[1]
    hdr['ox'] = -diam_m / 2.0
    hdr['oy'] = -diam_m / 2.0
    hdr['h'] = 0

    fits.writeto(f'Keck_ncpa_rmswfe{rms_wfe:.0f}nm.fits', opd, hdr, overwrite=True)
    

    return

def make_keck_pupil():
    """
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
    """
    inner_ring = 1
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

    # atlast.display(npix=1024, colorbar_orientation='vertical')

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

