import numpy as np
from pykoa.koa import Koa
from pathlib import Path
from astropy.table import Table, Column
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, LogStretch)

# By Brooke DiGia

def fetch_koa(date:str, koa_dir:Path, verbose:bool=False, download:bool=True):
    """
    Fetch KOA search results for an input date. Results table (.tbl)
    will be saved to koa_dir.

    Inputs:
    -------
    date    : str
        Date for which to search KOA

    koa_dir : Path
        Directory in which to save table of search results

    verbose : boolean, default=False
        Option to turn on verbose output

    Outputs:
    -------
    None. Saves filed titled {koa_dir}nirc2_search_{date}.tbl
    """
    # Download query table
    Koa.query_date('nirc2', f'{date}', koa_dir.as_posix() + f'nirc2_search_{date}.tbl', 
                   overwrite=True, format='ipac')
    # Read in query table
    rec = Table.read(koa_dir.as_posix() + f'nirc2_search_{date}.tbl', 
                     format='ipac')
    if verbose:
        print(rec)
        print(rec['koaimtyp'])

    if download:
        Koa.download(koa_dir.as_posix() + f'nirc2_search_{date}.tbl', 'ipac', \
        koa_dir.as_posix() + f'dnload_dir_nirc2_calib0_{date}', \
        start_row=0, \
        end_row=10, \
        lev1file=0, \
        calibfile=1, \
        calibdir=1)

def load_koa(koa_path:Path, plot:bool=True):
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
    koa_fits = fits.open(koa_path.as_posix())
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
    daofind = DAOStarFinder(fwhm=5.0, threshold=median + 30.0*std)
    sources = daofind(koa_img)  
    if verbose:
        sources.pprint()

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

        # Create interval object
        interval = MinMaxInterval()
        vmin, vmax = interval.get_limits(koa_img_sub)
        # Create an ImageNormalize object using a SqrtStretch object
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

        # Plot sky subtracted image
        fig, axes = plt.subplots()
        plot = axes.imshow(koa_img_sub, cmap="afmhot", norm=norm)
        plt.title(f"Sky-subtracted {koa_path.name}")
        plt.xlabel(f"{koa_img.shape[0]}")
        plt.xlabel(f"{koa_img.shape[1]}")
        cbar = fig.colorbar(plot, extend="max", shrink=0.7)
        cbar.minorticks_on()
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
    
    new_hdu = fits.PrimaryHDU(img_norm)
    return hdr, img_norm