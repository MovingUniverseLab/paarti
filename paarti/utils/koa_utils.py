import numpy as np
import pandas as pd
import maos_utils as mu
from pykoa.koa import Koa
from pathlib import Path
from astropy.table import Table
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize
from kai import instruments
import scipy
import astropy
import urllib
import os

# By Brooke DiGia

def check_mass_dimm(date_array:list, exp_array:list, saveto:str):
    """
    Function to check whether MASS/DIMM data exists for an input date
    range. 

    Inputs:
    -------
    date_array : list of strings
        List of dates to check MASS/DIMM for

        Format: YYYYMMDD

    exp_array  : list of strings
        List of exposure times of KOA images corresponding to 
        the date array (meaning ith element of exposure array
        should be the exposure of the image taken on ith date)

        Format: HH24:MI:SS

    saveto     : str
        Directory location in which to save the MASS/DIMM data,
        if it exists

    Outputs:
    --------

    
    """
    dimm_exists = np.empty(len(date_array))
    mass_exists = np.empty(len(date_array))
    data_exists = np.empty(len(date_array))
    for i, date in enumerate(date_array):
        # Check DIMM
        dimmdat = date + ".dimm.dat"
        url_root = "http://mkwc.ifa.hawaii.edu/current/seeing/"
        url = url_root + "dimm/" + dimmdat
        if not os.path.exists(saveto + dimmdat):
            try:
                # Pull and save DIMM file
                urllib.request.urlretrieve(url, saveto + dimmdat)
                print(f"{dimmdat} saved to {saveto}")
                dimm_exists[i] = True
            except Exception as error:
                print(f"Error while downloading {dimmdat} from {url}:", 
                      type(error).__name__, error)
                dimm_exists[i] = False
                data_exists[i] = False
        else:
            print(f"{dimmdat} exists in directory {saveto}, not downloading.")
            dimm_exists[i] = True

        # Check MASS
        masspro = date + ".masspro.dat"
        url = url_root + "masspro/" + masspro
        if not os.path.exists(saveto + masspro):
            try:
                # Pull and save MASS file
                urllib.request.urlretrieve(url, saveto + masspro)
                print(f"{masspro} saved to {saveto}")
                mass_exists[i] = True
            except Exception:
                print(f"Error while downloading {masspro} from {url}:", 
                      type(error).__name__, error)
                mass_exists[i] = False
                data_exists[i] = False
        else:
            print(f"{masspro} exists in directory {saveto}, not downloading.")
            mass_exists[i] = True
        
    
    return

def fetch_koa(start_date:str, end_date:str, koa_dir:Path, verbose:bool=False, download:bool=True):
    """
    Fetch KOA search results for an input date. Results table (.tbl)
    will be saved to koa_dir.

    Inputs:
    -------
    start_date : str
        Start date for which to search KOA

        Format: YYYY-MM-DD HH24:MI:SS

    end_date   : str
        End date for which to search KOA. Note that this can be the same
        as start_date as long as the actual timestamp is different.
        Same format as start_date

    koa_dir    : Path
        Directory in which to save table of search results

    verbose    : boolean, default=False
        Option to turn on verbose output

    Outputs:
    -------
    None. Saves filed titled {koa_dir}nirc2_search_{date}.tbl to KOA
    directory (koa_dir)
    """
    # Download query table (* is used to select all columns; 
    # 'filehand' and 'instrument' required for passing to Koa.download())
    query = f"select * from koa_nirc2 \
     where (utdatetime >= to_date('{start_date}', \
    'yyyy-mm-dd HH24:MI:SS') and \
     utdatetime <= to_date('{end_date}', 'yyyy-mm-dd HH24:MI:SS') \
     AND (targname='ao_confirmation' OR object='ao_confirmation'))"
    Koa.query_adql(query, koa_dir.as_posix() + f'nirc2_search_{start_date}_to_{end_date}.tbl', 
                   overwrite=True, format='ipac')
    # Read in query table
    rec = Table.read(koa_dir.as_posix() + f'nirc2_search_{start_date}_to_{end_date}.tbl', 
                     format='ipac')
    if verbose:
        print(rec)
    if download:
        Koa.download(koa_dir.as_posix() + f'nirc2_search_{start_date}_to_{end_date}.tbl', 'ipac', \
        koa_dir.as_posix() + f'dnload_dir_nirc2_calib0_{start_date}_to_{end_date}', \
        start_row=0, \
        end_row=10, \
        lev1file=0, \
        calibfile=1, \
        calibdir=1)
    return

def load_koa(koa_path:Path, plot:bool=False): # Default False for plotting since we plot in the sky subtraction function as well
    """
    Load in KOA FITS file

    Inputs:
    -------
    koa_path : Path
        Path to KOA FITS file to load in

    plot     : boolean, default=True
        Option to plot KOA image that was loaded in

    Outputs:
    --------
    koa_img  : 2D array, dtype=float
        KOA image data

    koa_hdr  : Astropy FITS header object
        KOA image header associated with data

    Plots KOA image
    """
    try:
        koa_fits = fits.open(koa_path.as_posix())
    except AttributeError as error:
        # koa_path is string, not Path object
        koa_fits = fits.open(koa_path)

    koa_hdu = koa_fits[0]
    koa_img = koa_fits[0].data
    koa_hdr = koa_fits[0].header

    # Purely for plotting purposes: 
    SMALL_SIZE = 10
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 13
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Show KOA image
    if plot:
        fig, axes = plt.subplots()
        plot = axes.imshow(koa_img, cmap="afmhot", vmin=0.0)
        plt.title(f"{koa_path.name}", fontsize=14)
        plt.xlabel(f"{koa_img.shape[0]}")
        plt.xlabel(f"{koa_img.shape[1]}")
        cbar = fig.colorbar(plot, extend="max", shrink=0.7)
        cbar.minorticks_on()
        plt.tight_layout()
        plt.show()

    return koa_img, koa_hdr

def centroid_koa(koa_img, verbose:bool=True):
    """
    Centroid sources within KOA image data. Returns coordinates of 
    brightest pixel (source)

    Inputs:
    -------
    koa_img : 2D array, dtype=float
        KOA image data

    verbose : boolean, default=True
        Option to turn on verbose output

    Outputs:
    --------
    coords  : 1D array of two elements, dtype=float
        Coordinates of brightest pixel in centroided source

    """
    # Calculate statistics on image prior to processing
    _, median, std = sigma_clipped_stats(koa_img, sigma=3.0) 
    # NIRC2/OSIRIS have FWHM of 5 pixels
    daofind = DAOStarFinder(fwhm=5.0, threshold=median + 50.0*std)
    sources = daofind(koa_img)  
    if verbose:
        try:
            sources.pprint()
        except AttributeError as error:
            print(f"Error printing sources: {error}. Likely no centroids were found by star finder.")
            return None

    # Find coordinates of brightest pixel
    coords = np.array([0.0, 0.0])
    coords[0] = sources['id'==np.argmax(sources['flux'])]['xcentroid']
    coords[1] = sources['id'==np.argmax(sources['flux'])]['ycentroid']
    return coords

def sky_subtract_koa(koa_img, koa_path:Path, source, r_in, r_out, verbose:bool=True):
    """
    Subtract sky background from KOA image

    Inputs:
    -------
    koa_img : 2D array, dtype=float
        KOA image data

    koa_path : Path
        Path to KOA FITS file to load in

    source  : 1D array of two entries, dtype=float
        Coordinates of source (brightest pixel in 
        koa_img)

    r_in    : int
        Inner radius in pixels for sky subtraction annulus

    r_out   : int
        Outer radius in pixels for sky subtraction annulus.
        Must be larger than r_in

    verbose : boolean, default=True
        Option to turn on verbose output

    Outputs:
    --------

    """
    sky_aperture = CircularAnnulus(source, r_in, r_out)
    sky_aper_photo = aperture_photometry(koa_img, sky_aperture)
    if verbose:
        print(sky_aper_photo)

    sky_sum = sky_aper_photo['aperture_sum'][0]
    sky_avg = sky_sum/sky_aperture.area
    koa_img_sub = koa_img - sky_avg
    if verbose:
        print(f"\nAverage pixel value in sky-subtracted KOA: {np.mean(koa_img_sub)}")
        print(f"Maximum pixel value in sky-subtracted KOA: {np.max(koa_img_sub)}")

        fig = plt.figure(figsize=(10.0, 10.0), layout='constrained')
        spec = fig.add_gridspec(1, 2)
        fig.suptitle("Original and sky-subtracted KOA images")
        ax0 = fig.add_subplot(spec[0,0])
        ax1 = fig.add_subplot(spec[0,1])

        # interval = MinMaxInterval()
        # vmin, vmax = interval.get_limits(koa_img)
        # # Create an ImageNormalize object using a SqrtStretch object
        # norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

        # Plot original image
        im0 = ax0.imshow(koa_img, cmap="inferno", vmin=0.0)
        try:
            ax0.set_title(f"{koa_path.name}")
        except AttributeError as error:
            # koa_path is string, not Path object
            ax0.set_title(f"{koa_path}")

        ax0.set_xlabel(f"{koa_img.shape[0]}")
        ax0.set_ylabel(f"{koa_img.shape[1]}")
        cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cbar0.minorticks_on()

        # Create interval object
        interval = MinMaxInterval()
        vmin, vmax = interval.get_limits(koa_img_sub)
        # Create an ImageNormalize object using a SqrtStretch object
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

        # Plot sky subtracted image
        im1 = ax1.imshow(koa_img_sub, cmap='inferno', norm=norm)
        try:
            ax1.set_title(f"Sky-subtracted {koa_path.name}")
        except AttributeError as error:
            # koa_path is string, not Path object
            ax1.set_title(f"Sky-subtracted {koa_path}")

        ax1.set_xlabel(f"{koa_img.shape[0]}")
        ax1.set_ylabel(f"{koa_img.shape[1]}")
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.minorticks_on()
        plt.show()

    return koa_img_sub

def clean_koa(koa_path, inner_radius:int=300, outer_radius:int=400):
    """
    Top-level function for processing KOA images prior to metric
    computation. 

    Inputs:
    -------
    koa_path        : Path
        Path to KOA FITS file to load in

    inner_radius    : int, default=300 px
        Inner radius in pixels for sky subtraction annulus

    outer_radius    : int, default=400 px
        Outer radius in pixels for sky subtraction annulus.
        Must be larger than r_in

    Outputs:
    --------
    new_hdu  : Astropy HDU
        Astropy HDU corresponding to new KOA image (sky-subtracted
        and cropped)

    """
    # Load in KOA image to process
    img, hdr = load_koa(koa_path)

    # Extract sources in image
    source_coords = centroid_koa(img)
    if source_coords is None:
        print(f"No sources found in KOA file {koa_path}, skip this image.")
        return None

    # Sky-subtract image
    img_sub = sky_subtract_koa(img, koa_path, source_coords, 
                               inner_radius, outer_radius)

    # Crop image around source
    cutout_obj = Cutout2D(img_sub, source_coords, [256, 256], mode='trim')
    # Normalize by integral of entire PSF (as opposed to peak pixel value)
    cropped_img = cutout_obj.data
    img_norm = cropped_img/np.sum(cropped_img)
    # Save normalization constant for each cropped KOA image to track outliers
    # (contributions from hot pixels, cosmic rays, etc.)
    with open('/Users/bdigia/myg3/data/KOA_norm_constants.csv', 'a+') as const_file:
        # Only write if this image's normalization has yet to be recorded
        to_write = f"{koa_path.name}: {np.sum(cropped_img)}\n"
        lines = const_file.readlines()
        for line in lines:
            if to_write in line:
                # Line is found in file, no need to inspect further or write
                # anything to file
                print("Already written to file.")
                break
        else:
            # Line was not found anywhere in file, so write it
            const_file.write(to_write)

    fig, axis = plt.subplots(figsize=(5.0, 5.0), layout='constrained')
    im0 = plt.imshow(cropped_img, cmap="inferno", vmin=0.0)
    plt.title(f"Cropped and sky-subtracted {koa_path.name}")
    plt.xlabel(f"{cropped_img.shape[0]}")
    plt.ylabel(f"{cropped_img.shape[1]}")
    cbar0 = plt.colorbar(im0, ax=axis, fraction=0.046, pad=0.04)
    cbar0.minorticks_on()
    plt.show()
    
    new_hdu = fits.PrimaryHDU(img_norm)
    return hdr, img_norm

def load_koa_starlist(starlist:Path):
    """
    Function to read in Keck star list text files

    Inputs:
    -------
    starlist : Path
        Path to star list .txt file

    Outputs:
    -------
    stardata : Pandas dataframe
        Dataframe containing information stored in starlist

    By Brooke DiGia
    """
    # Column specifications for starlist, as half-intervals
    colspecs = [(0, 3), (16, 18), (19, 21), (22, 28), (29, 32), (33, 35), (36, 41), (42, 48), (55, 59), (64, 68), (73, 74)]
    stardata = pd.read_fwf(starlist, colspecs=colspecs, header=None)
    stardata.columns = ['name', 'RA hh', 'RA mm', 'RA ss.sss', 'DEC +dd', 'DEC mm', 'DEC ss.ss', 'equinox', 'vmag', 'b-v', 'lgsflag']
    return stardata

def lookup_koa_object(koa_path:Path, starlist:Path):
    """
    Function to look up Keck engineering star in KOA eng image from star list(s).
    Keck star lists that are currently in this directory:
        - master_tycho_list_v12.txt
        - master_tycho_list_v10.txt

    Make sure you have these lists within your own working directory. They can be copied
    from the maos_utils module of the PAARTI repository

    Inputs:
    -------
    koa_path : Path
        Path to KOA FITS file to load in

    starlist : Path
        Path to star list .txt file

    Outputs:
    --------
    mag      : float
        V- or R-band magnitude of engineering star

    By Brooke DiGia     
    """
    # Load in KOA image
    _, hdr = load_koa(koa_path)

    # Load starlist
    engstars = load_koa_starlist(starlist)

    # Pull object + RA/DEC coordinates in degrees
    # (OBJECT keyword may not
    # name object observed, e.g. 'ao_confirmation')
    koa_object = hdr['OBJECT']
    koa_ra = hdr['RA']
    koa_ra_offset = hdr['RAOFF'] 
    koa_dec = hdr['DEC']
    koa_dec_offset = hdr['DECOFF']

    # FITS header defines above RA/DEC coordinates as that of telescope
    # and provides RA/DECOFF as right ascension and declination offsets 
    # --> adding these gives RA/DEC of object observed? Not entirely sure what
    # offset refers to
    koa_sky_coords = SkyCoord(ra=float(koa_ra+koa_ra_offset)*u.degree, 
                              dec=float(koa_dec+koa_dec_offset)*u.degree, 
                              frame='icrs')
    # Starlist has RA/DEC in hh/dd mm ss format
    koa_sky_coords.to_string('hmsdms')

    # Search engstars data for koa_object and pull corresponding
    # RA/DEC and magnitude (V or R band, depending on which is
    # available in starlist)
    mag = 0
    min_diff = np.inf
    min_diff_i = 0
    for i, star in engstars.iterrows():
        ra = f"{star['RA hh']}h{star['RA mm']}m{star['RA ss.sss']}s"
        dec = f"{star['DEC +dd']}h{star['DEC mm']}m{star['DEC ss.ss']}s"
        c = SkyCoord(ra, dec, frame='icrs')
        # Convert RA and DEC to degrees and subtract from target KOA
        # RA and DEC. Smallest difference's info will be stored
        ra_diff = koa_ra - c.ra
        dec_diff = koa_dec - c.dec
        # Add differences in quadrature
        diff = np.sqrt(ra_diff**2.0 + dec_diff**2.0)
        # If this difference is smaller than currently stored, replace
        # min_diff with difference
        if diff < min_diff:
            min_diff = diff
            min_diff_i = i
            print(star['vmag'])
            mag = star['vmag']

    return mag

"""
On-sky metric calculation routines from KAI copied and modified here for use on 
KOA images. The calculations all remain the same; the only KOA changes are related to
how the images are passed and loaded between the functions. The writing of the metrics
to an output text file has been removed. Sky subtraction flags on the image have been
removed since sky subtraction is automatically performed as part of KOA image
processing.
"""

def calc_strehl_on_sky(file_list, apersize=0.6, 
                       instrument=instruments.default_inst):
    """
    Calculate the Strehl, FWHM, and RMS WFE.
    The FWHM (and Strehl) is calculated over the specified
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
        The list of the KOA file names. These should be the ORIGINAL KOA images,
        as they will be cleaned, centroided, and cropped within this function
        as part of image processing (in lieu of AIROPA)

    out_file : str
        The name of the output text file.

    aper_size : float (def = 0.3 arcsec)
        The aperture size over which to calculate the Strehl and FWHM.

    skysub    : boolean (def = False)
        Option to perform sky subtraction on input PSF

    """
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
                                                         radius, skysub=True)
        # For each image, get the strehl, FWHM, RMS WFE, MJD, etc. and write to an
        # output file.
        strehls = []
        fwhms = []
        rmswfes = []
        empfwhms = []
        for ii in range(len(file_list)):
            strehl, fwhm, rmswfe, emp_fwhm = calc_strehl_single_on_sky(file_list[ii], radius, 
                                                                       dl_peak_flux_ratio, 
                                                                       skysub=False, # sky subtraction already performed by clean_koa, so do not perform again when passing KOA image
                                                                       instrument=instrument)
            strehls.append(strehl)
            fwhms.append(fwhm)
            rmswfes.append(rmswfe)
            empfwhms.append(emp_fwhm)
            mjd = fits.getval(file_list[ii], instrument.hdr_keys['mjd'])
    except astropy.nddata.PartialOverlapError:
        print("astropy.nddata.PartialOverlapError, failing gracefully...")
        strehls.append(-1.0)
        fwhms.append(-1.0)
        rmswfes.append(-1.0)
        empfwhms.append(-1.0)
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

    hdr, cropped_img = clean_koa(Path(img_file))

    # Position of Strehl source (cropping is done but Cutout2D centered on
    # source coords, so automatically we know coords to be center of image)
    coords = np.array([cropped_img.shape[0]/2.0, cropped_img.shape[1]/2.0])
    
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
        g2d = mu.fit_gaussian2d(cropped_img, coords, fwhm_boxsize*box_scale,
                             fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                             pos_delta_max=pos_delta_max, plot=True)
        sigma = (g2d.x_stddev_0.value + g2d.y_stddev_0.value) / 2.0
        fwhm = mu.stddev_to_fwhm(sigma)
        emp_fwhm = mu.empirical_fwhm(cropped_img, scale)

        # print(f"FWHM on iteration {iters} = {fwhm:.2f} mas | Empirical FWHM on iteration {iters} = {emp_fwhm:.2f} mas")

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
    peak_flux_ratio = calc_peak_flux_ratio_on_sky(cropped_img, coords, radius, skysub)

    # Normalize by the same from the DL image to get the Strehl.
    strehl = peak_flux_ratio / dl_peak_flux_ratio
    print('peak flux ratio = ', peak_flux_ratio, ' dl peak flux ratio = ', dl_peak_flux_ratio)

    # Convert the Strehl to a RMS WFE using the Marechal approximation.
    rms_wfe = np.sqrt( -1.0 * np.log(strehl)) * wavelength * 1.0e3 / (2. * np.pi)
    
    # Check final values and fail gracefully.
    if ((strehl < 0) or (strehl > 1) or
        (fwhm > 500) or (fwhm < (fwhm_min * scale * 1.0e3))):
        strehl = np.nan
        fwhm = np.nan
        rms_wfe = np.nan
    
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

    if skysub: # should not be turned on for KOA images, only DL images
        sky_rad_inn = radius + 20
        sky_rad_out = radius + 30
        sky_aper = CircularAnnulus(coords, sky_rad_inn, sky_rad_out)
        sky_aper_out = aperture_photometry(img, sky_aper)
        sky_aper_sum = sky_aper_out['aperture_sum'][0]

        aper_sum -= sky_aper_sum

    # Calculate the peak pixel flux / wide-aperture flux
    peak_flux_ratio = peak_flux / aper_sum
    
    return peak_flux_ratio