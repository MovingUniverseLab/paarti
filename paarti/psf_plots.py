from matplotlib import colors
import numpy as np
import pylab as plt


def grid_psf(psf, side=None, zoom=None, scale='log', color='range',
             linthresh=1e-4):
    """Show a regular square grid of PSFs.
    If the color coverage is 'zero' and the color scale is 'log', a symmetric
    logarithmic colormap normalization is used.

    Parameters:
        :param psf: PSF grid
        :type psf: psfs.PSF_stack
        :param side: Number of PSFs to show on each side of the grid (less than
            or equal to sqrt(n)). The first and last PSF of each column is
            always plotted. If 'None', all PSFs are shown.
        :type side: int
        :param zoom: Half-side of the region of a singular PSF to zoom in (").
            If 'None', the full PSF is shown.
        :type zoom: float, optional
        :param scale: Color scale ('linear' for linear, 'log' for logarithmic)
        :type scale: str, default 'log'
        :param color: Color coverage ('range' for the full range of values,
            'zero' for symmetric around 0). The first is useful to show a grid
            of PSFs, the second to show the difference of PSFs
        :type color: str, default 'range'
        :param  linthresh: Threshold to color PSF values as zero if a symmetric
            logarithmic scale is used
        :type  linthresh: float, default 1e-4
    """

    # Prepare grid
    n = psf.psfs.shape[0]

    if side is None:
        side = int(np.sqrt(n))

    n_side = int(np.sqrt(n))
    pos = np.linspace(0, (n_side - 1), side)
    pos = pos.astype(int)

    if zoom is None:
        zoom_px = round(psf.psfs.shape[1] / 2)
    else:
        zoom_px = np.ceil(zoom / psf.pixel_scale)

    zoom_px_min = (psf.psfs.shape[1] / 2) - zoom_px
    zoom_px_max = (psf.psfs.shape[1] / 2) + zoom_px

    # Calculate color range
    psf_color = psf.psfs[:, int(zoom_px_min): int(zoom_px_max),
                         int(zoom_px_min): int(zoom_px_max)]

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
            norm = colors.SymLogNorm(linthresh=linthresh, vmin=psf_color_min,
                                     vmax=psf_color_max)
        else:
            norm = colors.Normalize(vmin=psf_color_min, vmax=psf_color_max)

    # Plot
    plt.figure(figsize=(8, 9))

    for i_row in range(side):
        for i_col in range(side):
            idx = np.where((psf.pos[:, 0] == pos[i_row]) &
                           (psf.pos[:, 1] == pos[i_col]))[0][0]
            plt.subplot(side, side, (((side - i_row - 1) * side) +
                                     (i_col + 1)))
            plt.imshow(psf.psfs[idx, :, :], norm=norm,
                       cmap=plt.get_cmap(cmap_name), aspect='equal',
                       origin='lower')
            plt.xlim([zoom_px_min, zoom_px_max])
            plt.ylim([zoom_px_min, zoom_px_max])
            plt.tick_params(axis='x', bottom=False, labelbottom=False)
            plt.tick_params(axis='y', left=False, labelleft=False)

    plt.show(block=False)
