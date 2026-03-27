"""Functions to handle GIRMOS WFRs and PSFs.

References:
    [ref1] Ge et al., 1997
"""


import copy
import warnings

from matplotlib import animation, colors, patches, pyplot as plt
from mpl_toolkits import axes_grid1
import numpy as np
from os import path
from photutils import isophote
from scipy import io, linalg, ndimage, special

from source import misc


# Parameters
d = 8  # Telescope diameter (m)
d_o = 1.12  # Telescope central obstruction diameter (m)


def fwhm_double_gauss(i_c, w_c, i_h, w_h):
    """Calculate the FWHM of a double univariate Gaussians.

    Parameters:
        :param i_c: Peak of the core Gaussian
        :type i_c: float
        :param w_c: FWHM of the core Gaussian
        :type w_c: float
        :param i_h: Peak of the halo Gaussian
        :type i_h: float
        :param w_h: FWHM of the halo Gaussian
        :type w_h: float

    Returns:
        :return fwhm: FWHM
        :rtype fwhm: float
    """

    # Prepare the array
    r = np.arange(0, w_h, (w_h / 1e6))
    r2 = r ** 2
    f = (i_c * np.exp(-4 * np.log(2) * r2 / (w_c ** 2))) + \
        (i_h * np.exp(-4 * np.log(2) * r2 / (w_h ** 2)))  # [ref1], Eq. (19)
    f_diff = f - ((i_c + i_h) / 2)
    fwhm_min = r[np.where(f_diff >= 0)][-1]
    fwhm_max = r[np.where(f_diff <= 0)][0]
    fwhm = np.mean([fwhm_min, fwhm_max]) * 2

    return fwhm


def wfr_load(file, verbose=False):
    """Load a Matlab '.mat' file with the WFR cube into a variable.
    The aperture has a 128 x 128 px domain, with t temporal slices.

    Parameters:
        :param file: Filename, with path and extension
        :type file: string
        :param verbose: Print messages
        :type verbose: bool, default False

    Returns:
        :return wfr: WFR cube (m)
        :rtype wfr: numpy.ndarray [128, 128, t] (float)
    """

    # Load file
    if verbose:
        print("Loading WFRs ...")

    wfr_mat = io.loadmat(file)

    # Prepare the array
    wfr = wfr_mat['residue']

    return wfr


def wfr_plot(wfr, t_step=1, title="Wavefront", bar=True, save=None):
    """Plot a single temporal slice of a WFR cube.

    Parameters:
        :param wfr: WFR cube (m)
        :type wfr: numpy.ndarray [128, 128, t] (float)
        :param t_step: Index of the temporal slice (starting from 1)
        :type t_step: int, default 0
        :param title: Figure title
        :type title: str, default "Wavefront"
        :param bar: Show colorbar
        :type bar: bool, default True
        :param save: Path and filename of the saved image
        :type save: str, optional
    """

    # Calculate ticks
    wfr_t = wfr[:, :, (t_step - 1)] * 1e6
    d_px = wfr_t.shape[0]
    d_ticks_max = np.floor(d / 2)
    d_ticks = np.arange(-d_ticks_max, d_ticks_max, 1)
    d_ticks_str = [str(int(i)) for i in d_ticks]
    d_ticks_orig = (d_ticks * d_px / d) + ((d_px - 1) / 2)

    # Plot WFR
    fig = plt.figure(figsize=(5, 5.5))
    ax = fig.add_subplot(111)
    ax.set_position([0.15, 0.15, 0.7, 0.8])
    img = plt.imshow(wfr_t, cmap=plt.get_cmap('gray'), aspect='equal',
                     origin='lower')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xticks(d_ticks_orig)
    ax.set_yticks(d_ticks_orig)
    ax.set_xticklabels(d_ticks_str)
    ax.set_yticklabels(d_ticks_str)

    if bar:
        colorbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
        plt.colorbar(img, cax=colorbar_ax, orientation='horizontal',
                     label=r"Phase ($\mu m$)")

    plt.suptitle(title)
    plt.show(block=False)

    # Save image file
    if save is not None:
        plt.savefig(save)


def wfr_to_psf(wfr, wl, ps, psf_edges=None, psf_where=None, t_stack=100,
               verbose=False):
    """Transform a WFR into PSF.
    The final PSF is tip-tilt removed to te nearest integer pixel.

    Parameters:
        :param wfr: WFR cube (m)
        :type wfr: numpy.ndarray [128, 128, t] (float)
        :param wl: Wavelength (m)
        :type wl: float
        :param ps: PSF pixel scale ("/px)
        :type ps: float
        :param psf_edges: Left/right and top/bottom limits of the PSF (px)
        :type psf_edges: list [2] (int), optional
        :param psf_where: x and y coordinates where the PSF is defined
        :type psf_where: tuple [2] (numpy.ndarray (float)), optional
        :param t_stack: Number of PSF temporal slices to stack at a time, to
            reduce memory usage
        :type t_stack: int, default 100
        :param verbose: Print messages
        :type verbose: bool, default False

    Returns:
        :return psf_stack: Stacked PSF, with integral normalized to 1
        :rtype psf_stack: numpy.ndarray [x, x] (float)
        :return psf: PSF cube of the last 't_stack' slices, with the same
            normalization of 'psf_stack'
        :rtype psf: numpy.ndarray [x, x, t] (float)
        :param psf_edges: Left/right and top/bottom limits of the PSF (px)
        :type psf_edges: list [2] (int)
        :return psf_where: x and y coordinates where the PSF is defined
        :rtype psf_where: tuple [2] (numpy.ndarray (float))
    """

    # Prepare the aperture
    if verbose:
        print("Calculating PSF ...")

    wfr = 2 * np.pi * wfr / wl
    ap_ph = np.exp(-1j * wfr)
    ap_a = np.zeros(wfr.shape)
    ap_a[np.where(wfr != 0)] = 1
    ap_c = ap_a * ap_ph
    t_slices = ap_c.shape[2]

    # Pad the aperture
    pad = 206265 * wl * wfr.shape[0] / (ps * d)
    pad_add = int(round((pad - wfr.shape[0]) / 2))
    ap_pad_test = np.pad(ap_c[:, :, 0],
                         ((pad_add, pad_add), (pad_add, pad_add)), 'constant')
    pad_size = ap_pad_test.shape

    # Calculate PSF
    psf = np.zeros(np.array([pad_size[0], pad_size[1],
                             int(np.ceil(t_slices / t_stack))]))
    psf_tmp = []

    for i_t in range(t_slices):
        if verbose:
            print("\rCalculating time step # {0}/{1} ...".format((i_t + 1),
                                                                 t_slices),
                  end="")

        ap_pad = np.pad(ap_c[:, :, i_t], ((pad_add, pad_add),
                                          (pad_add, pad_add)), 'constant')

        if (i_t % t_stack) == 0:
            psf_tmp = np.zeros(np.array([pad_size[0], pad_size[1], t_stack]))

        psf_tmp[:, :, (i_t % t_stack)] = \
            np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.ifftshift(ap_pad))) ** 2)

        if ((i_t % t_stack) == (t_stack - 1)) | (i_t == (t_slices - 1)):
            psf[:, :, (i_t // t_stack)] = np.sum(psf_tmp, axis=2)

    if verbose:
        print()

    # Stack the PSF
    psf_stack = np.sum(psf, axis=2)

    # Define the PSF in a square
    if psf_edges is None:
        psf_center = np.argwhere(np.max(psf_stack) == psf_stack)[0][0]
        psf_r = np.min([(psf_stack.shape[0] - psf_center - 1), psf_center])
        psf_edges = [(psf_center - psf_r), (psf_center + psf_r + 1)]

    psf = psf[psf_edges[0]: psf_edges[1], psf_edges[0]: psf_edges[1], :]
    psf_stack = psf_stack[psf_edges[0]: psf_edges[1],
                          psf_edges[0]: psf_edges[1]]

    # Define the PSF in a circle
    if psf_where is None:
        psf_center_2 = (psf_stack.shape[0] - 1) / 2
        psf_space = np.linspace(0, (psf_stack.shape[0] - 1),
                                psf_stack.shape[0]) - psf_center_2
        psf_mesh_x, psf_mesh_y = np.meshgrid(psf_space, psf_space)
        psf_dist = np.hypot(psf_mesh_x, psf_mesh_y)
        psf_where = np.where(psf_dist <= psf_center_2)

    where_arr = np.zeros([psf_stack.shape[0], psf_stack.shape[0]])
    where_arr[psf_where] = 1
    notwhere = np.where(where_arr == 0)

    for i_t in range(psf.shape[2]):
        psf[:, :, i_t][notwhere] = 0

    psf_stack[notwhere] = 0

    # Normalize the PSF
    psf_stack_integral = np.sum(psf_stack)
    psf_stack = psf_stack / psf_stack_integral
    psf = psf / psf_stack_integral

    return psf_stack, psf, psf_edges, psf_where


def wfr_std(wfr, verbose=False):
    """Calculate the tip-tilt and high-order standard deviations of a stack of
    WFRs.

    Parameters:
        :param wfr: WFR cube (m)
        :type wfr: numpy.ndarray [128, 128, t] (float)
        :param verbose: Print messages
        :type verbose: bool, default False

    Returns:
        :return sigma: STD (m)
        :rtype sigma: float
        :return sigma_tt: Tip-tilt STD (rad)
        :rtype sigma_tt: float
        :return sigma_ho: High-order STD (m)
        :rtype sigma_ho: float
    """

    # Prepare the aperture
    if verbose:
        print("Calculating WFR variances ...")

    t_slices = wfr.shape[2]
    wfr_all_stds = np.zeros(t_slices)
    wfr_flat_stds = np.zeros(t_slices)
    tips_tilts = np.zeros([t_slices, 2])

    # Calculate STDs
    for i_t in range(t_slices):
        wfr_all_stds[i_t], _, wfr_flat_stds[i_t], tips_tilts[i_t, :] = \
            wfr_flatten(wfr[:, :, i_t])

    # Calculate STD
    sigma = np.mean(wfr_all_stds)

    # Calculate the tipt-tilt STD
    tips_tilts[:, 0] -= np.mean(tips_tilts[:, 0])
    tips_tilts[:, 1] -= np.mean(tips_tilts[:, 1])
    tts = np.hypot(tips_tilts[:, 0], tips_tilts[:, 1])
    # sigma_tt = np.std(tts)
    sigma_tt = np.mean(np.sqrt(np.linalg.eigvals(np.cov(np.transpose(tips_tilts)))))

    # Calculate the high_horder STD
    sigma_ho = np.mean(wfr_flat_stds)

    return sigma, sigma_tt, sigma_ho


def wfr_flatten(wfr):
    """Flatten a WFR by calculating and removing the tip-tilt. Calculate also
    the STD of the flattened WFR.

    Parameters:
        :param wfr: WFR (m)
        :type wfr: numpy.ndarray [128, 128, t] (float)

    Returns:
        :return wfr_all_std: WFR STD (m)
        :rtype wfr_all_std: float
        :return wfr_flat: Flattened WFR, with tip-tilt removed (m)
        :rtype wfr_flat: numpy.ndarray [128, 128, t] (float)
        :return wfr_flat_std: Flattened WFR STD (m)
        :rtype wfr_flat_std: float
        :return tip_tilt: Tip-tilt slopes (rad)
        :rtype tip_tilt: numpy.ndarray [2] (float)
    """

    # Flatten the WFR
    x, y = np.meshgrid(np.linspace(0, (wfr.shape[1] - 1), wfr.shape[1]),
                       np.linspace(0, (wfr.shape[0] - 1), wfr.shape[0]))
    wfr_where = np.where(wfr != 0)
    wfr_ok = wfr[wfr_where]
    wfr_all_std = np.std(wfr_ok)
    x_ok = x[wfr_where]
    y_ok = y[wfr_where]
    wfr_fit, _, _, _ = linalg.lstsq(np.c_[x_ok, y_ok, np.ones(len(wfr_ok))],
                                    wfr_ok)
    tip_tilt = np.array([wfr_fit[0], wfr_fit[1]])
    tip_tilt = np.arctan(tip_tilt * wfr.shape[0] / d)
    wfr_tip_tilt = (wfr_fit[0] * x_ok) + (wfr_fit[1] * y_ok) + wfr_fit[2]
    wfr_resid = wfr_ok - wfr_tip_tilt
    wfr_flat_std = np.std(wfr_resid)
    wfr_flat = copy.copy(wfr)
    wfr_flat[wfr_where] = wfr_resid

    return wfr_all_std, wfr_flat, wfr_flat_std, tip_tilt


def psf_plot(psf, ps, t_step=1, zoom=None, ticks=1, scale='log', color='range',
             color_range=None, psf_zero=1e-2, title="PSF", save=None,
             close=True):
    """Plot a single temporal slice of a PSF cube or the stacked PSF.
    The color scale can be linear or logarithmic. The plot can be saved as an
    image file.

    Parameters:
        :param psf: PSF cube or stacked PSF
        :type psf: numpy.ndarray [128, 128, t] or numpy.ndarray [128, 128]
            (float)
        :param ps: PSF pixel scale ("/px)
        :type ps: float
        :param t_step: Index of the temporal slice (starting from 1)
        :type t_step: int, default 0
        :param zoom: Half-side of the region to zoom in (")
        :type zoom: float, optional
        :param ticks: Axes' ticks distance (")
        :type ticks: float, default 1
        :param scale: Color scale ('linear' for linear, 'log' for logarithmic)
        :type scale: str, default 'log'
        :param color: Color coverage ('range' for a range of values, 'zero' for
            values symmetric around 0)
        :type color: str, default 'range'
        :param color_range: Color range ([min, max]). If not defined, it uses
            the maximum and minimum values
        :type color_range: np.ndarray [2] (float), optional
        :param  psf_zero: PSF values colored as zero if a symmetric logarithmic
            scale is used
        :type  psf_zero: float, default 1e-2
        :param title: Figure title
        :type title: str, default "PSF"
        :param save: Path and filename of the saved image
        :type save: str, optional
        :param close: Close the figure and don't show it
        :type close: bool, default True
    """

    # Select slice
    if len(psf.shape) == 3:
        psf_slice = psf[:, :, (t_step - 1)]
        slice_str = title + " (time step {0})".format(t_step)
    else:
        psf_slice = psf
        slice_str = title

    # Calculate zoom
    if zoom is None:
        zoom_px = psf.shape[0] / 2
    else:
        zoom_px = np.ceil(zoom / ps)

    zoom_px_min = (psf.shape[0] / 2) - zoom_px - 1
    zoom_px_max = (psf.shape[0] / 2) + zoom_px

    if not color_range:
        psf_color = psf_slice[int(zoom_px_min): int(zoom_px_max),
                              int(zoom_px_min): int(zoom_px_max)]
        color_range = [np.min(psf_color), np.max(psf_color)]

    if color == 'range':
        cmap_name = 'hot'

        if scale == 'log':
            norm = colors.LogNorm(vmin=color_range[0], vmax=color_range[1])
        else:
            norm = colors.Normalize(vmin=color_range[0], vmax=color_range[1])
    else:
        cmap_name = 'seismic'

        if scale == 'log':
            norm = colors.SymLogNorm(linthresh=psf_zero, vmin=color_range[0],
                                     vmax=color_range[1])
        else:
            norm = colors.Normalize(vmin=color_range[0], vmax=color_range[1])

    # Plot PSF
    x_ticks_orig, y_ticks_orig, x_ticks_str, y_ticks_str = \
        misc.ticks_scale(psf_slice.shape, ps, ticks)
    fig = plt.figure(figsize=(5, 5.5))
    ax = fig.add_subplot(111)
    ax.set_position([0.15, 0.15, 0.7, 0.8])
    img = plt.imshow(psf_slice, norm=norm, cmap=plt.get_cmap(cmap_name),
                     aspect='equal', origin='lower')
    ax.set_xlabel("x (\")")
    ax.set_ylabel("y (\")")
    ax.set_xticks(x_ticks_orig)
    ax.set_yticks(y_ticks_orig)
    ax.set_xticklabels(x_ticks_str)
    ax.set_yticklabels(y_ticks_str)
    plt.xlim([zoom_px_min, zoom_px_max])
    plt.ylim([zoom_px_min, zoom_px_max])
    colorbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
    cb = plt.colorbar(img, cax=colorbar_ax, orientation='horizontal',
                      label="Counts")
    cb.ax.tick_params(labelsize=10)
    fig.suptitle(slice_str)
    plt.show(block=False)

    # Save image file
    if save is not None:
        plt.savefig(save)

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def psf_rad(psf, psf_where, ps, t_step=1, zoom=None, title="PSF", save=None):
    """Radial plot of a single temporal slice of a PSF cube or the stacked PSF.
    The plot can be saved as an image file.

    Parameters:
        :param psf: PSF cube or stacked PSF
        :type psf: numpy.ndarray [128, 128, t] or numpy.ndarray [128, 128]
            (float)
        :param psf_where: x and y coordinates where the PSF is defined
        :type psf_where: tuple [2] (numpy.ndarray (float))
        :param ps: PSF pixel scale ("/px)
        :type ps: float
        :param t_step: Index of the temporal slice (starting from 1)
        :type t_step: int, default 0
        :param zoom: Maximum radius to zoom in (")
        :type zoom: float, optional
        :param title: Figure title
        :type title: str, default "PSF"
        :param save: Path and filename of the saved image
        :type save: str, optional
    """

    # Select slice
    if len(psf.shape) == 3:
        psf_slice = psf[:, :, (t_step - 1)]
        slice_str = title + " (time step {0})".format(t_step)
    else:
        psf_slice = psf
        slice_str = title

    # Calculate distances
    psf_center = (psf.shape[0] - 1) / 2
    psf_space = np.linspace(0, (psf.shape[0] - 1), psf.shape[0]) - psf_center
    psf_mesh_x, psf_mesh_y = np.meshgrid(psf_space, psf_space)
    psf_dist = np.hypot(psf_mesh_x, psf_mesh_y) * ps

    # Prepare plotted values
    x_plot = psf_dist[psf_where].flatten()
    y_plot = psf_slice[psf_where].flatten()

    # Calculate zoom
    if zoom is None:
        zoom = np.max(psf_dist)

    # Plot PSF
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.plot(x_plot, y_plot, '.', markersize=1, color='k')
    ax.set_xlabel("r (\")")
    ax.set_ylabel("I")
    plt.xlim([0, zoom])
    fig.suptitle(slice_str)
    plt.show(block=False)

    # Save image file
    if save is not None:
        plt.savefig(save)


def psf_movie(psf, ps, zoom=None, ticks=0.1, interval=20, scale='log',
              verbose=False):
    """Plot the movie of a PSF cube or the stacked PSF.
    The color scale can be linear or logarithmic.

    Parameters:
        :param psf: PSF cube
        :type psf: numpy.ndarray [128, 128, t] (float)
        :param ps: PSF pixel scale ("/px)
        :type ps: float
        :param zoom: Half-side of the region to zoom in (")
        :type zoom: float, optional
        :param ticks: Axes' ticks distance (")
        :type ticks: float, default 0.1
        :param interval: Time interval between frames (ms)
        :type interval: float, default 20
        :param scale: Color scale ('linear' for linear, 'log' for logarithmic)
        :type scale: str, default 'log'
        :param verbose: Print messages
        :type verbose: bool, default False
    """

    # Calculate zoom
    if verbose:
        print("Plotting movie ...")

    if zoom is None:
        zoom_px = psf.shape[0] / 2
    else:
        zoom_px = np.ceil(zoom / ps)

    zoom_px_min = (psf.shape[0] / 2) - zoom_px
    zoom_px_max = (psf.shape[0] / 2) + zoom_px

    # Calculate color range
    psf_color = psf[int((psf.shape[1] / 2) - zoom_px):
                    int((psf.shape[1] / 2) + zoom_px),
                    int((psf.shape[0] / 2) - zoom_px):
                    int((psf.shape[0] / 2) + zoom_px), :]
    psf_color_min = np.percentile(psf_color, 20)
    psf_color_max = np.max(psf_color)

    # Prepare figure
    x_ticks_orig, y_ticks_orig, x_ticks_str, y_ticks_str = \
        misc.ticks_scale(psf.shape, ps, ticks)
    norm = colors.Normalize

    if scale == 'log':
        norm = colors.LogNorm

    imgs = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_position([0.15, 0.15, 0.7, 0.7])
    ax.set_xlabel("x (\")")
    ax.set_ylabel("y (\")")
    ax.set_xticks(x_ticks_orig)
    ax.set_yticks(y_ticks_orig)
    ax.set_xticklabels(x_ticks_str)
    ax.set_yticklabels(y_ticks_str)

    # Generate frames
    for i_frame in range(psf.shape[2]):
        if verbose:
            print("\rPreparing frame # {0}/{1} ...".format((i_frame + 1),
                                                           psf.shape[2]),
                  end="")

        img = ax.imshow(psf[:, :, i_frame],
                        norm=norm(vmin=psf_color_min, vmax=psf_color_max),
                        cmap=plt.get_cmap('jet'), aspect='equal',
                        origin='lower', animated=True)
        plt.xlim([zoom_px_min, zoom_px_max])
        plt.ylim([zoom_px_min, zoom_px_max])
        txt = ax.text(0.5, 0.85,
                      "   Normalized PSF (frame # {0})   ".format(i_frame + 1),
                      ha='center', transform=fig.transFigure, fontsize=14,
                      bbox=dict(boxstyle='square', fc='w', ec='w'))
        imgs.append([img, txt])

    if verbose:
        print()

    # Play movie
    animation.ArtistAnimation(fig, imgs, interval=interval, blit=True,
                              repeat_delay=1000)
    plt.show(block=False)


def psf_metrics(wfr, wl, ps, over=5, psf_edges=None, psf_where=None,
                psf_edges_over=None, psf_where_over=None, verbose=False):
    """Measure PSF metrics using an oversampled PSF.

    Parameters:
        :param wfr: WFR cube (m)
        :type wfr: numpy.ndarray [128, 128, t] (float)
        :param wl: Wavelength (m)
        :type wl: float
        :param ps: PSF pixel scale ("/px)
        :type ps: float
        :param over: Oversampling of the PSF pixel scale
        :type over: float, default 5
        :param psf_edges: left/right and top/bottom limits of the PSF (px)
        :type psf_edges: list [2] (int), optional
        :param psf_where: x and y coordinates where the PSF is defined
        :type psf_where: tuple [2] (numpy.ndarray (float)), optional
        :param psf_edges_over: left/right and top/bottom limits of the
            oversampled PSF (px)
        :type psf_edges_over: list [2] (int), optional
        :param psf_where_over: x and y coordinates where the oversampled PSF is
            defined
        :type psf_where_over: tuple [2] (numpy.ndarray (float)), optional
        :param verbose: Print messages
        :type verbose: bool, default False

    Returns:
        :return metr: PSF metrics
        :rtype metr: dict
            :dreturn sr: Strehl ratio
            :dtype sr: float
            :deturn fwhm_maj: FWHM along major axis (")
            :dtype fwhm_maj: float
            :deturn fwhm_min: FWHM along minor axis (")
            :dtype fwhm_min: float
            :deturn fwhm_ell: FWHM ellipticity
                ((fwhm_max - fwhm_min) / fwhm_max)
            :dtype fwhm_ell: float
            :deturn fwhm_a: FWHM arithmetic mean (")
            :dtype fwhm_a: float
            :deturn fwhm_g: FWHM geometric mean (")
            :dtype fwhm_g: float
            :deturn fwhm_pa: FWHM position angle (measured from the positive x
                axis, towards positive y) (deg)
            :dtype fwhm_pa: float
            :deturn ens_rad: Ensquared energy radii (")
            :dtype ens_rad: np.ndarray (float) [t]
            :deturn ens: Ensquared energy curve
            :dtype ens: np.ndarray (float) [t]
            :deturn enc_rad: Encircled energy radii (")
            :dtype enc_rad: np.ndarray (float) [t]
            :deturn enc: Encircled energy curve
            :dtype enc: np.ndarray (float) [t]
            :deturn ens25: 25% ensquared energy (")
            :dtype ens25: float
            :deturn ens50: 50% ensquared energy (")
            :dtype ens50: float
            :deturn ens80: 80% ensquared energy (")
            :dtype ens80: float
            :deturn enc25: 25% encircled energy (")
            :dtype enc25: float
            :deturn enc50: 50% encircled energy (")
            :dtype enc50: float
            :deturn enc80: 80% encircled energy (")
            :dtype enc80: float
            :deturn ens0025: Fraction of the ensquared energy in 0.025"
            :dtype ens0025: float
            :deturn ens005: Fraction of the ensquared energy in 0.05"
            :dtype ens005: float
            :deturn ens01: Fraction of the ensquared energy in 0.1"
            :dtype ens01: float
            :deturn enc0025: Fraction of the encircled energy in 0.025"
            :dtype enc0025: float
            :deturn enc005: Fraction of the encircled energy in 0.05"
            :dtype enc005: float
            :deturn enc01: Fraction of the encircled energy in 0.1"
            :dtype enc01: float
            :deturn nea: Noise equivalent area ("^2)
            :dtype nea: float
        :return psf: PSF
        :rtype psf: numpy.ndarray [a, a] (float)
        :return psf_xy: PSF positions (x, y) (")
        :rtype psf_xy: numpy.ndarray [a, a, 2] (float)
        :param psf_edges: left/right and top/bottom limits of the PSF (px)
        :type psf_edges: list [2] (int)
        :return psf_where: x and y coordinates where the PSF is defined
        :rtype psf_where: tuple [2] (numpy.ndarray (float))
        :return psf_over: Oversampled PSF
        :rtype psf_over: numpy.ndarray [b, b] (float)
        :return psf_xy_over: Oversampled PSF positions (x, y) (")
        :rtype psf_xy_over: numpy.ndarray [b, b, 2] (float)
        :param psf_edges_over: left/right and top/bottom limits of the
            oversampled PSF (px)
        :type psf_edges_over: list [2] (int)
        :return psf_where_over: x and y coordinates where the oversampled PSF is
            defined
        :rtype psf_where_over: tuple [2] (numpy.ndarray (float))
        :return ps_over: Oversampled PSF pixel scale ("/px)
        :rtype ps_over: float
    """

    # Generate the PSF
    psf, _, psf_edges, psf_where = wfr_to_psf(wfr, wl, ps, psf_edges=psf_edges,
                                              psf_where=psf_where,
                                              verbose=verbose)
    x_ee0, y_ee0 = \
        np.meshgrid(np.linspace(0, (psf.shape[1] - 1), psf.shape[1]),
                    np.linspace(0, (psf.shape[0] - 1), psf.shape[0]))
    center_mass0 = ndimage.measurements.center_of_mass(psf)
    x_ee0 = x_ee0 - center_mass0[1]
    y_ee0 = y_ee0 - center_mass0[0]
    psf_xy = np.dstack((x_ee0, y_ee0))
    psf_xy *= ps

    # Generate the oversampled PSF
    if verbose:
        print("Calculating PSF metrics ...")

    ps_over = ps / over
    psf_over, _, psf_edges_over, psf_where_over = \
        wfr_to_psf(wfr, wl, ps_over, psf_edges=psf_edges_over,
                   psf_where=psf_where_over, verbose=verbose)

    # Generate the diffraction-limited PSF
    wfr_flat = wfr[:, :, 0]
    wfr_flat[np.where(wfr_flat != 0)] = 1
    wfr_flat = wfr_flat[:, :, np.newaxis]
    psf_diffr, _, _, _ = wfr_to_psf(wfr_flat, wl, ps_over,
                                    psf_edges=psf_edges_over,
                                    psf_where=psf_where_over, verbose=verbose)

    # Calculate SR
    peak_over = np.max(psf_over)
    peak_diffr = np.max(psf_diffr)
    sr = peak_over / peak_diffr

    # Calculate FWHM
    contour_zoom = over * 5
    psf_ell = psf_over[(int(psf_over.shape[0] / 2) - contour_zoom):
                       (int(psf_over.shape[0] / 2) + contour_zoom),
                       (int(psf_over.shape[1] / 2) - contour_zoom):
                       (int(psf_over.shape[1] / 2) + contour_zoom)]
    pa_guess = [0.01, 0.5, 1, 1.5, 2, 2.5, 3]
    eps_guess = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    sma_guess = [over, (over * 2)]
    eps_pa_sma_guess = [[i_pa, i_eps, i_sma] for i_sma in sma_guess for i_eps in
                        eps_guess for i_pa in pa_guess]
    eps_pa_n = 0
    ell_iso = []
    hm = peak_over / 2
    isos_same = True

    while ((len(ell_iso) == 0) or isos_same) and \
            (eps_pa_n < len(eps_pa_sma_guess)):
        try:
            ell_guess = \
                isophote.EllipseGeometry(x0=contour_zoom, y0=contour_zoom,
                                         sma=eps_pa_sma_guess[eps_pa_n][2],
                                         eps=eps_pa_sma_guess[eps_pa_n][1],
                                         pa=eps_pa_sma_guess[eps_pa_n][0])
            ell_fit = isophote.Ellipse(psf_ell, ell_guess)
            warnings.simplefilter('ignore')
            ell_iso = ell_fit.fit_image()
            warnings.simplefilter('default')

            if len(ell_iso) > 0:
                ell_table_tmp = ell_iso.to_table()
                isos_diff_tmp = ell_table_tmp['intens'].data - hm
                isos_opposite = isos_diff_tmp[0] * isos_diff_tmp[-1]

                if isos_opposite < 0:
                    isos_same = False
        except (ValueError, IndexError):
            pass

        eps_pa_n += 1

    ell_table = ell_iso.to_table()

    if len(ell_table) > 0:
        isos_diff = ell_table['intens'].data - hm
        isos_idx1 = len(np.where(isos_diff >= 0)[0])
        isos_idx2 = isos_idx1 - 1
        isos_1 = ell_table['intens'][isos_idx1]
        isos_2 = ell_table['intens'][isos_idx2]
        isos_diff1 = hm - isos_1
        isos_diff2 = isos_2 - isos_1
        fwhm_maj = (ell_table['sma'][isos_idx1] +
                    (isos_diff1 *
                     (ell_table['sma'][isos_idx2] - ell_table['sma'][isos_idx1])
                     / isos_diff2)) * 2 * ps_over
        fwhm_ell = ell_table['ellipticity'][isos_idx1] + \
            (isos_diff1 * (ell_table['ellipticity'][isos_idx2] -
                           ell_table['ellipticity'][isos_idx1]) / isos_diff2)
        fwhm_min = fwhm_maj - (fwhm_ell * fwhm_maj)
        fwhm_a = (fwhm_maj + fwhm_min) / 2
        fwhm_g = np.sqrt(fwhm_maj * fwhm_min)
        fwhm_pa_tmp = ell_table['pa'][isos_idx1] + \
            (isos_diff1 * (ell_table['pa'][isos_idx2] -
                           ell_table['pa'][isos_idx1]) / isos_diff2)
        fwhm_pa = fwhm_pa_tmp.value
    else:
        fwhm_maj = np.nan
        fwhm_ell = np.nan
        fwhm_min = np.nan
        fwhm_a = np.nan
        fwhm_g = np.nan
        fwhm_pa = np.nan

    # Calculate EEs
    ee_step = 0.005
    ee_buffer_px = 10
    ee_max_px = (psf_over.shape[0] / 2) - ee_buffer_px
    ens_max_arcsec = ee_max_px * ps_over / np.sqrt(2)
    enc_max_arcsec = ee_max_px * ps_over
    ens_rads_arcsec = np.arange(0, ens_max_arcsec, ee_step)
    enc_rads_arcsec = np.arange(0, enc_max_arcsec, ee_step)
    ens_rads_px = ens_rads_arcsec / ps_over
    enc_rads_px = enc_rads_arcsec / ps_over
    x_ee, y_ee = \
        np.meshgrid(np.linspace(0, (psf_over.shape[1] - 1), psf_over.shape[1]),
                    np.linspace(0, (psf_over.shape[0] - 1), psf_over.shape[0]))
    center_mass = ndimage.measurements.center_of_mass(psf_over)
    x_ee = x_ee - center_mass[1]
    y_ee = y_ee - center_mass[0]
    psf_xy_over = np.dstack((x_ee, y_ee))
    psf_xy_over *= ps_over
    r_ee = np.hypot(x_ee, y_ee)
    ens_where_max = np.where(r_ee <= ens_rads_px[-1])
    ens_max = np.sum(psf_over[ens_where_max])
    enc_where_max = np.where(r_ee <= enc_rads_px[-1])
    enc_max = np.sum(psf_over[enc_where_max])
    ens = np.zeros(len(ens_rads_px))
    enc = np.zeros(len(enc_rads_px))

    for i_rad in range(len(ens_rads_px)):
        if verbose:
            print(("\rCalculating ensquared energy #" +
                   " {0}/{1} ...").format((i_rad + 1), len(ens_rads_px)),
                  end="")

        ens_rad_px = ens_rads_px[i_rad]
        ens_where = np.where((x_ee <= ens_rad_px) & (x_ee >= -ens_rad_px) &
                             (y_ee <= ens_rad_px) & (y_ee >= -ens_rad_px))
        ens[i_rad] = np.sum(psf_over[ens_where]) / ens_max

    for i_rad in range(len(enc_rads_px)):
        if verbose:
            print(("\rCalculating encirlcled energy #" +
                   " {0}/{1} ...").format((i_rad + 1), len(enc_rads_px)),
                  end="")

        enc_rad_px = enc_rads_px[i_rad]
        enc_where = np.where(r_ee <= enc_rad_px)
        enc[i_rad] = np.sum(psf_over[enc_where]) / enc_max

    if verbose:
        print()

    ens25_diff = ens - 0.25
    ens25_idx2 = len(np.where(ens25_diff <= 0)[0])
    ens25_idx1 = ens25_idx2 - 1
    ens25_1 = ens25_diff[ens25_idx1]
    ens25_2 = ens25_diff[ens25_idx2]
    ens25 = (ens_rads_arcsec[ens25_idx1] +
             (-ens25_1 *
              (ens_rads_arcsec[ens25_idx2] - ens_rads_arcsec[ens25_idx1]) /
              (ens25_2 - ens25_1)))
    ens50_diff = ens - 0.5
    ens50_idx2 = len(np.where(ens50_diff <= 0)[0])
    ens50_idx1 = ens50_idx2 - 1
    ens50_1 = ens50_diff[ens50_idx1]
    ens50_2 = ens50_diff[ens50_idx2]
    ens50 = (ens_rads_arcsec[ens50_idx1] +
             (-ens50_1 *
              (ens_rads_arcsec[ens50_idx2] - ens_rads_arcsec[ens50_idx1]) /
              (ens50_2 - ens50_1)))
    ens80_diff = ens - 0.8
    ens80_idx2 = len(np.where(ens80_diff <= 0)[0])
    ens80_idx1 = ens80_idx2 - 1
    ens80_1 = ens80_diff[ens80_idx1]
    ens80_2 = ens80_diff[ens80_idx2]
    ens80 = (ens_rads_arcsec[ens80_idx1] +
             (-ens80_1 *
              (ens_rads_arcsec[ens80_idx2] - ens_rads_arcsec[ens80_idx1]) /
              (ens80_2 - ens80_1)))
    enc25_diff = enc - 0.25
    enc25_idx2 = len(np.where(enc25_diff <= 0)[0])
    enc25_idx1 = enc25_idx2 - 1
    enc25_1 = enc25_diff[enc25_idx1]
    enc25_2 = enc25_diff[enc25_idx2]
    enc25 = (enc_rads_arcsec[enc25_idx1] +
             (-enc25_1 *
              (enc_rads_arcsec[enc25_idx2] - enc_rads_arcsec[enc25_idx1]) /
              (enc25_2 - enc25_1)))
    enc50_diff = enc - 0.5
    enc50_idx2 = len(np.where(enc50_diff <= 0)[0])
    enc50_idx1 = enc50_idx2 - 1
    enc50_1 = enc50_diff[enc50_idx1]
    enc50_2 = enc50_diff[enc50_idx2]
    enc50 = (enc_rads_arcsec[enc50_idx1] +
             (-enc50_1 *
              (enc_rads_arcsec[enc50_idx2] - enc_rads_arcsec[enc50_idx1]) /
              (enc50_2 - enc50_1)))
    enc80_diff = enc - 0.8
    enc80_idx2 = len(np.where(enc80_diff <= 0)[0])
    enc80_idx1 = enc80_idx2 - 1
    enc80_1 = enc80_diff[enc80_idx1]
    enc80_2 = enc80_diff[enc80_idx2]
    enc80 = (enc_rads_arcsec[enc80_idx1] +
             (-enc80_1 *
              (enc_rads_arcsec[enc80_idx2] - enc_rads_arcsec[enc80_idx1]) /
              (enc80_2 - enc80_1)))
    ens_rad0025 = np.where(ens_rads_arcsec == 0.025)[0][0]
    ens_rad005 = np.where(ens_rads_arcsec == 0.05)[0][0]
    ens_rad01 = np.where(ens_rads_arcsec == 0.1)[0][0]
    enc_rad0025 = np.where(enc_rads_arcsec == 0.025)[0][0]
    enc_rad005 = np.where(enc_rads_arcsec == 0.05)[0][0]
    enc_rad01 = np.where(enc_rads_arcsec == 0.1)[0][0]
    ens0025 = ens[ens_rad0025]
    ens005 = ens[ens_rad005]
    ens01 = ens[ens_rad01]
    enc0025 = enc[enc_rad0025]
    enc005 = enc[enc_rad005]
    enc01 = enc[enc_rad01]

    # Calculate NEA
    psf_2 = psf_over ** 2
    nea = 1 / (np.sum(psf_2) * (over ** 2))
    nea *= ps ** 2

    # Prepare output
    metr = {'sr': sr, 'fwhm_maj': fwhm_maj, 'fwhm_min': fwhm_min,
            'fwhm_a': fwhm_a, 'fwhm_g': fwhm_g, 'fwhm_ell': fwhm_ell,
            'fwhm_pa': fwhm_pa, 'ens_rad': ens_rads_arcsec, 'ens': ens,
            'enc_rad': enc_rads_arcsec, 'enc': enc, 'ens25': ens25,
            'ens50': ens50, 'ens80': ens80, 'enc25': enc25, 'enc50': enc50,
            'enc80': enc80, 'ens0025': ens0025, 'ens005': ens005,
            'ens01': ens01, 'enc0025': enc0025, 'enc005': enc005,
            'enc01': enc01, 'nea': nea}

    return metr, psf, psf_xy, psf_edges, psf_where, psf_over, psf_xy_over,\
        psf_edges_over, psf_where_over, ps_over


def psf_diff(psfs, psf_where, idx_ref):
    """Compare PSFs by calculating the residual respect to a reference one, and
    then measuring its STD and integral.

    The STD is measured on the ratio between the residual and the reference PSF.
    The resulting value describes the typical variation of a pixel from the
    reference PSF.

    The integral is measured on the absolute of the residual. The resulting
    value describes the fraction of a PSF that is different from the reference
    one.

    Parameters:
        :param psfs: Stack of PSFs
        :type psfs: numpy.ndarray [x, x, n] (float)
        :param psf_where: x and y coordinates where the PSF is defined
        :type psf_where: tuple [2] (numpy.ndarray (float))
        :param idx_ref: Index of the reference PSF
        :type idx_ref: int

    Returns:
        :return psf_res_std: STD of the PSF residual
        :rtype psf_res_std: numpy.ndarray [n] (float)
        :return psf_res_sum: Sum of the absolute PSF residual
        :rtype psf_res_sum: numpy.ndarray [n] (float)
        :return psf_res: stack of residual PSFs
        :rtype psf_res: numpy.ndarray [x, x, n] (float)
    """

    # Select the reference PSF
    psf_ref = psfs[:, :, idx_ref]

    # Compare PSFs
    n_psf = psfs.shape[2]
    psf_res = np.zeros(psfs.shape)
    psf_res_std = np.zeros(n_psf)
    psf_res_sum = np.zeros(n_psf)

    for i_psf in range(n_psf):

        if i_psf != idx_ref:
            psf_res_i = psfs[:, :, i_psf] - psf_ref
            psf_res[:, :, i_psf] = psf_res_i
            psf_res_std[i_psf] = np.std(psf_res_i[psf_where] /
                                        psf_ref[psf_where])
            psf_res_sum[i_psf] = np.sum(np.abs(psf_res_i))

    return psf_res_std, psf_res_sum, psf_res


def en_plot(en, rad, title="Encircled energy", save=None, close=True):
    """Plot an enclosed energy curve, either ensquared or encircled.

    Parameters:
        :param en: Enclosed energy curve
        :type en: np.ndarray (float) [t]
        :param rad: Enclosed energy radii (")
        :type rad: np.ndarray (float) [t]
        :param title: Figure title
        :type title: str, default "Encircled energy"
        :param save: Path and filename of the saved image
        :type save: str, optional
        :param close: Close the figure and don't show it
        :type close: bool, default True
    """

    # Plot curve
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    plt.plot(rad, en)
    ax.set_xlabel("Radius (\")")
    ax.set_ylabel("Fraction of enclosed energy")
    plt.title(title)
    plt.show(block=False)

    # Save plot
    plt.savefig(save)

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def grid_plot(grid, max_range_pos, tick_step, regard_r, name,
              filename, folder, range_val=None, units=None, color_r=False,
              close=True):
    """Plot a grid of PSF parameters.

    Parameters:
        :param grid: Grid of values
        :type grid: np.ndarray (float) [side, side]
        :param max_range_pos: Maximum range in x and y of the PSF positions (")
        :type max_range_pos: int
        :param tick_step: Axes' tick step (")
        :type tick_step: float
        :param regard_r: Field of regard radius (px)
        :type regard_r: float
        :param name: Name of the PSF parameter
        :type name: str
        :param filename: Filename of the saved figure
        :type filename: str
        :param folder: Parent of the folder containing the saved figures
        :type folder: str
        :param range_val: Range in values to plot [min, max]
        :type range_val: list [2] (float), optional
        :param units: Units of the parameter
        :type units: str, optional
        :param color_r: Reverse colormap
        :type color_r: bool; default False
        :param close: Close the figure and don't show it
        :type close: bool, default True

    Returns:
        :return ax_pos: Axes position
        :rtype ax_pos: Bbox
    """

    # Calculate ticks
    n_side = grid.shape[0]
    ticks = np.arange(-max_range_pos, (max_range_pos + tick_step), tick_step)
    ticks_pos = (ticks + max_range_pos) * (n_side - 1) / (2 * max_range_pos)
    ticks.astype(str)

    # Plot grid
    _, ax = plt.subplots(figsize=(6, 5))
    im = plt.imshow(grid, origin='lower')

    if range_val is not None:
        im.set_clim(vmin=range_val[0], vmax=range_val[1])

    circ = patches.Circle((((n_side - 1) / 2), ((n_side - 1) / 2)),
                          radius=regard_r, ls='--', linewidth=2,
                          edgecolor='b', facecolor='none')
    ax.add_patch(circ)
    ax.set_position([0.12, 0.1, 0.75, 0.75])

    if color_r:
        cmap_name = 'hot_r'
        cmin = 0.1
        cmax = 1
    else:
        cmap_name = 'hot'
        cmin = 0
        cmax = 0.9

    cmap = plt.get_cmap(cmap_name)
    c_select = cmap(np.linspace(cmin, cmax, cmap.N))
    cmap_new = colors.LinearSegmentedColormap.from_list(cmap_name, c_select)
    plt.set_cmap(cmap_new)
    plt.xticks(ticks_pos, ticks)
    plt.yticks(ticks_pos, ticks)
    ax.set_xlabel("x (\")", fontsize=15)
    ax.set_ylabel("y (\")", fontsize=15)
    plt.title(name, y=1.05, fontsize=18)
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.7)
    clb = plt.colorbar(im, cax=cax)

    if units:
        clb.ax.set_title(units, fontsize=15)

    # Save plot
    plt.savefig(path.join(folder, (filename + '.png')))

    # Record axes positions
    ax_pos = plt.get(ax, 'position')

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)

    return ax_pos


def grid_angle_plot(x, y, angles, lengths, max_range, tick_step, regard_r,
                    scale, ax_pos, name, filename, folder, close=True):
    """Plot a grid of PSF angular parameters.

    Parameters:
        :param x: Grid of x positions (")
        :type x: np.ndarray (float) [n]
        :param y: Grid of y positions (")
        :type y: np.ndarray (float) [n]
        :param angles: Grid of angles
        :type angles: np.ndarray (float) [n]
        :param lengths: Lengths of the arrows
        :type lengths: np.ndarray (float) [n]
        :param max_range: Maximum range in x and y of the PSF positions (")
        :type max_range: int
        :param tick_step: Axes' tick step (")
        :type tick_step: float
        :param regard_r: Field of regard radius (")
        :type regard_r: float
        :param scale: Ellipticity plot scale
        :type scale: float
        :param ax_pos: Axes position
        :type ax_pos: Bbox
        :param name: Name of the PSF parameter
        :type name: str
        :param filename: Filename of the saved figure
        :type filename: str
        :param folder: Parent of the folder containing the saved figures
        :type folder: str
        :param close: Close the figure and don't show it
        :type close: bool, default True
    """

    # # Calculate ticks
    ticks_pos = np.arange(-max_range, (max_range + tick_step), tick_step)
    ticks = ticks_pos
    ticks.astype(str)

    # Plot grid
    _, ax = plt.subplots(figsize=(6, 5))
    u = np.cos(np.deg2rad(angles)) * lengths
    v = np.sin(np.deg2rad(angles)) * lengths
    quiv = plt.quiver(x, y, u, v, scale_units='xy', scale=(1 / scale),
                      headlength=0, headaxislength=0)
    plt.quiver(x, y, -u, -v, scale_units='xy', scale=(1 / scale), headlength=0,
               headaxislength=0)
    circ = patches.Circle((0, 0), radius=regard_r, ls='--', linewidth=2,
                          edgecolor='b', facecolor='none')
    ax.add_patch(circ)
    ax.set_position([0.12, 0.1, 0.75, 0.75])
    ax_lim = (max_range * 2) / (np.sqrt(len(x)) - 1) / 2
    ax.set_xlim([-(max_range + ax_lim), (max_range + ax_lim)])
    ax.set_ylim([-(max_range + ax_lim), (max_range + ax_lim)])
    ax.set_position(ax_pos)
    plt.xticks(ticks_pos, ticks)
    plt.yticks(ticks_pos, ticks)
    ax.set_xlabel("x (\")", fontsize=15)
    ax.set_ylabel("y (\")", fontsize=15)
    plt.title(name, y=1.05, fontsize=18)
    plt.quiverkey(quiv, (1.5 * max_range), (0.8 * max_range), 0.2,
                  "0.1 ellipticity", coordinates='data')

    # Save plot
    plt.savefig(path.join(folder, (filename + '.png')))

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def grid_psf_plot(stack, col, row, ps, side=None, zoom=None, scale='log',
                  color='range', psf_zero=1e-2, title="PSF", save=None,
                  close=True):
    """Plot a grid of PSFs.
    If the color coverage is 'zero' and the color scale is 'log', a symmetric
    logarithmic colormap normalization is used.

    Parameters:
        :param stack: Stack of PSFs
        :type stack: np.ndarray (float) [px, px, n]
        :param col: Column indeces
        :type col: np.ndarray (int) [n]
        :param row: Row indeces
        :type row: np.ndarray (int) [n]
        :param ps: PSF pixel scale ("/px)
        :type ps: float
        :param side: Number of PSFs to show on the side of the grid (less than
            or equal to sqrt(n))
        :type side: int
        :param zoom: Half-side of the region to zoom in (")
        :type zoom: float, optional
        :param scale: Color scale ('linear' for linear, 'log' for logarithmic)
        :type scale: str, default 'log'
        :param color: Color coverage ('range' for the full range of values,
            'zero' for symmetric around 0)
        :type color: str, default 'range'
        :param  psf_zero: PSF values colored as zero if a symmetric logarithmic
            scale is used
        :type  psf_zero: float, default 1e-2
        :param title: Figure title
        :type title: str, default "PSF"
        :param save: Path and filename of the saved image
        :type save: str, optional
        :param close: Close the figure and don't show it
        :type close: bool, default True
    """

    # Prepare grid
    n = stack.shape[2]

    if side is None:
        side = int(np.sqrt(n))

    n_side = int(np.sqrt(n))
    pos = np.linspace(0, (n_side - 1), side)
    pos = pos.astype(int)

    if zoom is None:
        zoom_px = stack.shape[0] / 2
    else:
        zoom_px = np.ceil(zoom / ps)

    zoom_px_min = (stack.shape[0] / 2) - zoom_px
    zoom_px_max = (stack.shape[0] / 2) + zoom_px

    # Calculate color range
    psf_color = stack[int(zoom_px_min): int(zoom_px_max),
                      int(zoom_px_min): int(zoom_px_max), :]

    if color == 'range':
        cmap_name = 'hot'
        psf_color_min = np.min(psf_color)
        psf_color_max = np.max(psf_color)

        if scale == 'log':
            norm = colors.LogNorm(vmin=psf_color_min, vmax=psf_color_max)
        else:
            norm = colors.Normalize(vmin=psf_color_min, vmax=psf_color_max)
    else:
        cmap_name = 'seismic'
        psf_color_max = np.max(np.abs(psf_color))
        psf_color_min = -psf_color_max

        if scale == 'log':
            norm = colors.SymLogNorm(linthresh=psf_zero, vmin=psf_color_min,
                                     vmax=psf_color_max)
        else:
            norm = colors.Normalize(vmin=psf_color_min, vmax=psf_color_max)

    # Plot the figure
    fig = plt.figure(figsize=(8, 9))

    for i_row in range(side):
        for i_col in range(side):
            idx = np.where((col == pos[i_col]) & (row == pos[i_row]))[0][0]
            plt.subplot(side, side, (((side - i_row - 1) * side) + (i_col + 1)))
            plt.imshow(stack[:, :, idx], norm=norm,
                       cmap=plt.get_cmap(cmap_name), aspect='equal',
                       origin='lower')
            plt.xlim([zoom_px_min, zoom_px_max])
            plt.ylim([zoom_px_min, zoom_px_max])
            plt.tick_params(axis='x', bottom=False, labelbottom=False)
            plt.tick_params(axis='y', left=False, labelleft=False)

    fig.suptitle(title, fontsize=20)
    plt.show(block=False)

    # Save image file
    if save is not None:
        plt.savefig(save)

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def stats_plot(grid, dist, regard_r, label, frmt, name, filename, folder,
               max_val=None, units="", ratio=False, close=True):
    """Calculate and plot statistics of PSF parameters.

    Parameters:
        :param grid: Grid of values
        :type grid: np.ndarray (float) [side, side]
        :param dist: Grid of distances from the center of the field of view (")
        :type dist: np.ndarray (float) [side, side]
        :param regard_r: Field of regard radius ("px")
        :type regard_r: float
        :param label: Label of the PSF parameter (without units)
        :type label: str
        :param frmt: Format of the statistics of the PSF parameter
        :type frmt: str
        :param name: Name of the PSF parameter
        :type name: str
        :param filename: Filename of the saved figure
        :type filename: str
        :param folder: Parent of the folder containing the saved figures
        :type folder: str
        :param max_val: Maximum value to plot
        :type max_val: float, optional
        :param units: Units of the PSF parameter
        :type units: str, default ""
        :param ratio: Plotting ratio values
        :type ratio: bool, default False
        :param close: Close the figure and don't show it
        :type close: bool, default True

    Returns:
        :return ax_pos: Axes position
        :rtype ax_pos: Bbox
    """

    # Calculate statistics
    regard = np.where(dist <= regard_r)
    grid_regard = grid[regard]
    grid_min = np.min(grid_regard)
    grid_max = np.max(grid_regard)
    grid_mean = np.mean(grid_regard)
    grid_median = np.median(grid_regard)

    # Plot statistics
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.set_position([0.12, 0.12, 0.5, 0.7])
    plt.scatter(dist, grid, c='k', s=4)
    ylim = plt.ylim()

    if plt.ylim()[0] >= 0:
        if max_val:
            ylim = [0, max_val]
        else:
            ylim = [0, (1.1 * plt.ylim()[1])]
    else:
        if plt.ylim()[1] <= 0:
            if max_val:
                ylim = [(1.1 * plt.ylim()[0]), max_val]
            else:
                ylim = [(1.1 * plt.ylim()[0]), 0]

    xlim = plt.xlim()
    plt.plot(([regard_r] * 2), ylim, '--b', alpha=0.5)

    if ratio:
        plt.plot(xlim, ([1] * 2), '--r', alpha=0.5)

    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_xlabel("r (\")", fontsize=15)

    if units == "":
        ylabel = label
    else:
        ylabel = (label + " (" + units + ")")

    ax.set_ylabel(ylabel, fontsize=15)
    plt.title(name, y=1.05, fontsize=18)
    plt.text(1.1, 0.7, ("Minimum: {0:" + frmt + "} {1}").format(grid_min,
                                                                units),
             transform=ax.transAxes, fontsize=13)
    plt.text(1.1, 0.55, ("Maximum: {0:" + frmt + "} {1}").format(grid_max,
                                                                 units),
             transform=ax.transAxes, fontsize=13)
    plt.text(1.1, 0.4, ("Mean: {0:" + frmt + "} {1}").format(grid_mean, units),
             transform=ax.transAxes, fontsize=13)
    plt.text(1.1, 0.25, ("Median: {0:" + frmt + "} {1}").format(grid_median,
                                                                units),
             transform=ax.transAxes, fontsize=13)

    # Save plot
    plt.savefig(path.join(folder, (filename + '.png')))

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def double_stats_plot(grid1, grid2, dist, regard_r, label, label1, label2, name,
                      filename, folder, diffr_lim=None, max_val=None, units="",
                      close=True):
    """Plot statistics of two PSF parameters.

    Parameters:
        :param grid1: First grid of values
        :type grid1: np.ndarray (float) [side, side]
        :param grid2: Second grid of values
        :type grid2: np.ndarray (float) [side, side]
        :param dist: Grid of distances from the center of the field of view (")
        :type dist: np.ndarray (float) [side, side]
        :param regard_r: Field of regard radius ("px")
        :type regard_r: float
        param label: Label of the y axis (without units)
        :type label: str
        :param label1: Label of the first PSF parameter
        :type label1: str
        :param label2: Label of the second PSF parameter
        :type label2: str
        :param name: Name of the PSF parameter
        :type name: str
        :param filename: Filename of the saved figure
        :type filename: str
        :param folder: Parent of the folder containing the saved figures
        :type folder: str
        :param diffr_lim: Diffraction limit FWHM to plot
        :type diffr_lim: float, optional
        :param max_val: Maximum value to plot
        :type max_val: float, optional
        :param units: Units of the PSF parameter
        :type units: str, default ""
        :param close: Close the figure and don't show it
        :type close: bool, default True

    Returns:
        :return ax_pos: Axes position
        :rtype ax_pos: Bbox
    """

    # Plot statistics
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_position([0.12, 0.12, 0.7, 0.7])
    plt.scatter(dist, grid1, marker='o', c='b', s=5, label=label1)
    plt.scatter(dist, grid2, marker='o', c='r', s=5, label=label2)
    ylim = plt.ylim()

    if plt.ylim()[0] >= 0:
        if max_val:
            ylim = [0, max_val]
        else:
            ylim = [0, (1.1 * plt.ylim()[1])]
    else:
        if plt.ylim()[1] <= 0:
            if max_val:
                ylim = [(1.1 * plt.ylim()[0]), max_val]
            else:
                ylim = [(1.1 * plt.ylim()[0]), 0]

    xlim = plt.xlim()
    plt.plot(([regard_r] * 2), ylim, '--b', alpha=0.5)

    if diffr_lim:
        plt.plot(xlim, ([diffr_lim] * 2), '--r', alpha=0.5,
                 label="Diffraction limit")

    plt.legend(loc='lower left')
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_xlabel("r (\")", fontsize=15)

    if units == "":
        ylabel = label
    else:
        ylabel = (label + " (" + units + ")")

    ax.set_ylabel(ylabel, fontsize=15)
    plt.title(name, y=1.05, fontsize=18)

    # Save plot
    plt.savefig(path.join(folder, (filename + '.png')))

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def triple_stats_plot(grid1, grid2, grid3, dist, regard_r, label, label1,
                      label2, label3, name, filename, folder, diffr_lim=None,
                      max_val=None, units="", close=True):
    """Plot statistics of three PSF parameters.

    Parameters:
        :param grid1: First grid of values
        :type grid1: np.ndarray (float) [side, side]
        :param grid2: Second grid of values
        :type grid2: np.ndarray (float) [side, side]
        :param grid3: Third grid of values
        :type grid3: np.ndarray (float) [side, side]
        :param dist: Grid of distances from the center of the field of view (")
        :type dist: np.ndarray (float) [side, side]
        :param regard_r: Field of regard radius ("px")
        :type regard_r: float
        param label: Label of the y axis (without units)
        :type label: str
        :param label1: Label of the first PSF parameter
        :type label1: str
        :param label2: Label of the second PSF parameter
        :type label2: str
        :param label3: Label of the third PSF parameter
        :type label3: str
        :param name: Name of the PSF parameter
        :type name: str
        :param filename: Filename of the saved figure
        :type filename: str
        :param folder: Parent of the folder containing the saved figures
        :type folder: str
        :param diffr_lim: Diffraction limit FWHM to plot
        :type diffr_lim: float, optional
        :param max_val: Maximum value to plot
        :type max_val: float, optional
        :param units: Units of the PSF parameter
        :type units: str, default ""
        :param close: Close the figure and don't show it
        :type close: bool, default True

    Returns:
        :return ax_pos: Axes position
        :rtype ax_pos: Bbox
    """

    # Plot statistics
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_position([0.12, 0.12, 0.7, 0.7])
    plt.scatter(dist, grid1, marker='o', c='b', s=5, label=label1)
    plt.scatter(dist, grid2, marker='o', c='r', s=5, label=label2)
    plt.scatter(dist, grid3, marker='o', c='orange', s=5, label=label3)
    ylim = plt.ylim()

    if plt.ylim()[0] >= 0:
        if max_val:
            ylim = [0, max_val]
        else:
            ylim = [0, (1.1 * plt.ylim()[1])]
    else:
        if plt.ylim()[1] <= 0:
            if max_val:
                ylim = [(1.1 * plt.ylim()[0]), max_val]
            else:
                ylim = [(1.1 * plt.ylim()[0]), 0]

    xlim = plt.xlim()
    plt.plot(([regard_r] * 2), ylim, '--b', alpha=0.5)

    if diffr_lim:
        plt.plot(xlim, ([diffr_lim] * 2), '--r', alpha=0.5,
                 label="Diffraction limit")

    plt.legend(loc='lower left')
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_xlabel("r (\")", fontsize=15)

    if units == "":
        ylabel = label
    else:
        ylabel = (label + " (" + units + ")")

    ax.set_ylabel(ylabel, fontsize=15)
    plt.title(name, y=1.05, fontsize=18)

    # Save plot
    plt.savefig(path.join(folder, (filename + '.png')))

    # Close the image
    if close:
        plt.close()
    else:
        plt.show(block=False)


def interpolate_dir(psf_in, direct):
    """Calculate the GIRMOS PSF in a direction, using the bilinear interpolation
    of the PSF grid.

    Parameters:
        :param psf_in: GIRMOS PSF grid
        :type psf_in: dict
        :param direct: Direction of the observation respect to the optical axis
            ([x, y]) (")
        :type direct: list [2] (float)

    Returns:
        :return psf_out: Interpolated PSF
        :rtype psf_out: numpy.ndarray [m, n] (float)
    """

    # Find nearby PSFs
    print("Interpolating PSF ...")
    dr = np.hypot(psf_in['x'] - direct[0], psf_in['y'] - direct[1])

    # Find PSFs in each quadrant
    xlo_ylo = np.where((psf_in['x'] <= direct[0]) &
                       (psf_in['y'] <= direct[1]))[0]
    xlo_yhi = np.where((psf_in['x'] <= direct[0]) &
                       (psf_in['y'] > direct[1]))[0]
    xhi_ylo = np.where((psf_in['x'] > direct[0]) &
                       (psf_in['y'] <= direct[1]))[0]
    xhi_yhi = np.where((psf_in['x'] > direct[0]) &
                       (psf_in['y'] > direct[1]))[0]

    # Find the closest PSF in each quadrant
    idx_xlo_ylo = xlo_ylo[np.argmin(dr[xlo_ylo])]
    idx_xlo_yhi = xlo_yhi[np.argmin(dr[xlo_yhi])]
    idx_xhi_ylo = xhi_ylo[np.argmin(dr[xhi_ylo])]
    idx_xhi_yhi = xhi_yhi[np.argmin(dr[xhi_yhi])]

    # Select the the closest PSF in each quadrant
    psf_xlo_ylo = psf_in['psfs_arr'][:, :, idx_xlo_ylo]
    psf_xlo_yhi = psf_in['psfs_arr'][:, :, idx_xlo_yhi]
    psf_xhi_ylo = psf_in['psfs_arr'][:, :, idx_xhi_ylo]
    psf_xhi_yhi = psf_in['psfs_arr'][:, :, idx_xhi_yhi]

    # Calculate the x distance of the closest PSF in each quadrant
    dx_xlo_ylo = np.abs(psf_in['x'][idx_xlo_ylo] - direct[0])
    dx_xlo_yhi = np.abs(psf_in['x'][idx_xlo_yhi] - direct[0])
    dx_xhi_ylo = np.abs(psf_in['x'][idx_xhi_ylo] - direct[0])
    dx_xhi_yhi = np.abs(psf_in['x'][idx_xhi_yhi] - direct[0])

    # Calculate the y distance of the closest PSF in each quadrant
    dy_xlo_ylo = np.abs(psf_in['y'][idx_xlo_ylo] - direct[1])
    dy_xlo_yhi = np.abs(psf_in['y'][idx_xlo_yhi] - direct[1])
    dy_xhi_ylo = np.abs(psf_in['y'][idx_xhi_ylo] - direct[1])
    dy_xhi_yhi = np.abs(psf_in['y'][idx_xhi_yhi] - direct[1])

    # Calculate the sides of the PSFs asterism
    dx_bottom = psf_in['x'][idx_xhi_ylo] - psf_in['x'][idx_xlo_ylo]
    dx_top = psf_in['x'][idx_xhi_yhi] - psf_in['x'][idx_xlo_yhi]
    dy_left = psf_in['y'][idx_xlo_yhi] - psf_in['y'][idx_xlo_ylo]
    dy_right = psf_in['y'][idx_xhi_yhi] - psf_in['y'][idx_xhi_ylo]

    # Calculate the weight of the PSFs
    weight_xlo_ylo = (1 - (dx_xlo_ylo / dx_bottom)) * \
                     (1 - (dy_xlo_ylo / dy_left))
    weight_xlo_yhi = (1 - (dx_xlo_yhi / dx_top)) * (1 - (dy_xlo_yhi / dy_left))
    weight_xhi_ylo = (1 - (dx_xhi_ylo / dx_bottom)) * \
                     (1 - (dy_xhi_ylo / dy_right))
    weight_xhi_yhi = (1 - (dx_xhi_yhi / dx_top)) * (1 - (dy_xhi_yhi / dy_right))

    # Interpolate PSF
    psf_out = (weight_xlo_ylo * psf_xlo_ylo) + \
              (weight_xlo_yhi * psf_xlo_yhi) + \
              (weight_xhi_ylo * psf_xhi_ylo) + \
              (weight_xhi_yhi * psf_xhi_yhi)

    return psf_out


def co_fwhm(verbose=True):
    """Calculate the central obscuration factor for the diffraction-limited
    FWHM of the Gemini telescope. It can also prints the value.

    Parameters:
        :param verbose: Print messages
        :type verbose: bool, default True

    Returns:
        :return fwhm_co: FWHM central obscuration factor
        :rtype fwhm_co: float
    """

    # Prepare arrays
    co_w = d_o / d
    cos = [0, co_w]
    fwhma = np.zeros(2)
    r = np.arange(0.01, 10, 0.01)

    # Calculate FWHMa
    for i_co in range(2):
        psfun = (((2 * special.j1(r) / r) -
                  (2 * cos[i_co] * special.j1(cos[i_co] * r) / r)) /
                 (1 - (cos[i_co] ** 2))) ** 2
        fwhm_pre = np.where(psfun >= 0.5)[0][-1]
        r_pre = r[fwhm_pre]
        r_post = r[fwhm_pre + 1]
        psfun_pre = psfun[fwhm_pre]
        psfun_post = psfun[fwhm_pre + 1]
        m_fwhm = (psfun_post - psfun_pre) / (r_post - r_pre)
        fwhma[i_co] = 2 * (((0.5 - psfun_pre) / m_fwhm) + r_pre)

    # Calculate the central obstruction factor
    fwhm_co = fwhma[1] / fwhma[0]

    if verbose:
        print("FWHM central obscuration correction: {0:.3f}".format(fwhm_co))

    return fwhm_co
