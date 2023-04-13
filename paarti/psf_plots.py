from matplotlib import colors
import numpy as np
import pylab as plt


def grid_psf(psf, side=None, zoom=None, scale='log', color='range',
             linthresh=1e-4):
    """Show a regular square grid of PSFs.
    If the color coverage is 'zero' and the color scale is 'log', a symmetric
    logarithmic colormap normalization is used.

    Inputs:
    -------
    psf: psfs.PSF_stack
        PSF grid

    Optional Inputs:
    ----------------
    side: int or None
        Number of PSFs to show on each side of the grid (less than
        or equal to sqrt(n)). The first and last PSF of each column is
        always plotted. If 'None', all PSFs are shown.
    zoom: float or None
        Size of PSF box, in arcsec. If 'None', the full PSF is shown.
    scale: str
        Color scale. 'linear' for linear, 'log' for logarithmic
    color: str
        Color coverage ('range' for the full range of values,
        'zero' for symmetric around 0). The first is useful to show a grid
        of PSFs, the second to show the difference of PSFs.
    linthresh: float
        Threshold to color PSF values as zero if a symmetric
        logarithmic scale is used.
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


def plot_psf_stack(psf_stack, zoom=None,
                   scale='log', color='range', linthresh=1e-4):
    """Show a regular square grid of PSFs.
    If the color coverage is 'zero' and the color scale is 'log', a symmetric
    logarithmic colormap normalization is used.

    Inputs:
    -------
    psf: psfs.PSF_stack
        PSF grid

    Optional Inputs:
    ----------------
    side: int or None
        Number of PSFs to show on each side of the grid (less than
        or equal to sqrt(n)). The first and last PSF of each column is
        always plotted. If 'None', all PSFs are shown.
    zoom: float or None
        Size of PSF box, in arcsec. If 'None', the full PSF is shown.
    scale: str
        Color scale. 'linear' for linear, 'log' for logarithmic
    color: str
        Color coverage ('range' for the full range of values,
        'zero' for symmetric around 0). The first is useful to show a grid
        of PSFs, the second to show the difference of PSFs.
    linthresh: float
        Threshold to color PSF values as zero if a symmetric
        logarithmic scale is used.
    """

    # Prepare grid
    n = psf_stack.psfs.shape[0]
    pos = psf_stack.pos

    zoom_px = round(psf_stack.psfs.shape[1] / 2)
    if ((zoom is not None) and
        ((zoom / psf_stack.pixel_scale) < zoom_px)):
        zoom_px = np.ceil(zoom / psf_stack.pixel_scale)

    zoom_px_min = (psf_stack.psfs.shape[1] / 2) - zoom_px
    zoom_px_max = (psf_stack.psfs.shape[1] / 2) + zoom_px

    # Calculate color range
    psf_color = psf_stack.psfs[:,
                               int(zoom_px_min): int(zoom_px_max),
                               int(zoom_px_min): int(zoom_px_max)]

    if color == 'range':
        cmap_name = 'hot_r'
        psf_color_min = np.min(psf_color)
        psf_color_max = np.max(psf_color)

        if scale == 'log':
            norm = colors.LogNorm(vmin=psf_color_min, vmax=psf_color_max)
        else:
            norm = colors.Normalize(vmin=psf_color_min, vmax=psf_color_max)
    else:
        cmap_name = 'seismic_r'
        psf_color_max = np.max(np.abs(psf_color))
        psf_color_min = -psf_color_max

        if scale == 'log':
            norm = colors.SymLogNorm(linthresh=linthresh, vmin=psf_color_min,
                                     vmax=psf_color_max)
        else:
            norm = colors.Normalize(vmin=psf_color_min, vmax=psf_color_max)

            
    # Figure out how big we should make each PSF.
    # We will simply use the sqrt of the number of PSFs to
    # approximate. This works well for a square grid. 
    n_psfs_side = int(n**0.5)

    # Setup the boundaries of the PSF boxes (leave room for axis labels).
    plot_box_size = 0.85
    xlo = 0.1
    xhi = xlo + plot_box_size
    ylo = 0.05
    yhi = ylo + plot_box_size

    box_scale = 0.95
    psf_axes_size = (plot_box_size / n_psfs_side) * box_scale
    psf_box_size = n_psfs_side * psf_axes_size # does not include padding
    scale_param = n_psfs_side / (n_psfs_side + 2)

    # Figure out the max and min positions on the sky to
    # fit in all the PSFs.
    xpos_rng = pos[:, 0].max() - pos[:, 0].min()
    ypos_rng = pos[:, 1].max() - pos[:, 1].min()
    xpos_cen = pos[:, 0].mean()
    ypos_cen = pos[:, 1].mean()
    pos_rng = np.max([xpos_rng, ypos_rng])
    pos_box_size = pos_rng * plot_box_size * box_scale / ((n_psfs_side - 1) * psf_axes_size)

    # Convert positions into figure coordinates. These will mark
    # the center of the PSF boxes.
    xpos_min = xpos_cen - 0.5*pos_box_size
    ypos_min = ypos_cen - 0.5*pos_box_size
    
    xpos_min_fig = xpos_min * (plot_box_size / pos_box_size)
    xpos_min_fig += xlo
    ypos_min_fig = ypos_min * (plot_box_size / pos_box_size)
    ypos_min_fig += ylo

    xpos_fig = (pos[:, 0] - xpos_min) * plot_box_size / pos_box_size
    xpos_fig += xlo
    ypos_fig = (pos[:, 1] - ypos_min) * plot_box_size / pos_box_size
    ypos_fig += ylo
    
    # Plot
    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()

    for pp in range(len(pos)):
        xfig = xpos_fig[pp]
        yfig = ypos_fig[pp]

        # Need to figure out the lower left corner (rather than center)
        # of this box. 
        xfig_min = xfig - 0.5 * psf_axes_size
        yfig_min = yfig - 0.5 * psf_axes_size
        
        ax = fig.add_axes([xfig_min, yfig_min, psf_axes_size, psf_axes_size])
        ax.imshow(psf_stack.psfs[pp, :, :], norm=norm,
                       cmap=plt.get_cmap(cmap_name), aspect='equal',
                       origin='lower')

        ax.set_xlim([zoom_px_min, zoom_px_max])
        ax.set_ylim([zoom_px_min, zoom_px_max])
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)

        if psf_stack.metrics:
            fwhm = psf_stack.metrics['emp_fwhm'][pp]
            ellip = psf_stack.metrics['ellipticity'][pp]
            strehl = psf_stack.metrics['strehl'][pp]

            str = f'FWHM = {fwhm*1e3:.0f} mas\n'
            str += f'ellip = {ellip:.2f}\n'
            str += f'strehl = {strehl*100:.0f}%\n'
            
            ax.text(0.02, 0.98, str,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=12)
            

    # Make the axis labels.
    lab_x = np.unique(pos[:, 0])
    lab_y = np.unique(pos[:, 1])
    lab_x_fig = (lab_x - xpos_min) * plot_box_size / pos_box_size
    lab_x_fig += xlo
    lab_y_fig = (lab_y - ypos_min) * plot_box_size / pos_box_size
    lab_y_fig += ylo

    for xx in range(len(lab_x)):
        fig.text(lab_x_fig[xx], yhi, f'{lab_x[xx]:5.1f}',
                 ha='center', va='bottom')
    for yy in range(len(lab_y)):
        fig.text(xlo, lab_y_fig[yy], f'{lab_y[yy]:5.1f}',
                 ha='right', va='center', rotation='vertical')

    fig.text(xlo + 0.5 * plot_box_size, 0.95, 'East - West',
             ha='center',
             weight='bold', fontsize=24)
    fig.text(0.01, ylo + 0.5 * plot_box_size, 'South - North',
             va='center', rotation='vertical',
             weight='bold', fontsize=24)

        
    plt.show(block=False)

    return
