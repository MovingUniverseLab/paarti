# quant_map_vis.py
# ---
# Script to visualize PSF quantity (e.g. Strehl) on a map
# Authors: Abhimat Gautam

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree

def quant_map_vis(plot_quant, plot_x, plot_y,
                  refstars=None, guidestar,
                  ttstar=None,
                  plot_type='Voronoi',
                  plot_quant_label='',
                  plot_out_file_name='quant_map',
                  plot_color_lims=None,
                  plot_color_cmap='viridis',
                  plot_mag_map='Wistia'): #default = viridis
    # Set up Voronoi plot, and bounds
    x_max = np.max(plot_x)
    x_min = np.min(plot_x)
    x_range = x_max - x_min
    
    y_max = np.max(plot_y)
    y_min = np.min(plot_y)
    y_range = y_max - y_min

    #x_max = np.max(plot_x)-512 #Change from pixel-space to arcsec offset-space.
    #x_min = np.min(plot_x)-512
    #x_range = x_max - x_min
    
    #y_max = np.max(plot_y)-512
    #y_min = np.min(plot_y)-512
    #y_range = y_max - y_min
    
    #plot_XYs = np.stack((plot_x, plot_y), axis=-1) #pixel-scale
    plot_XYs = np.stack((plot_x-512, plot_y-512), axis=-1) #arcsec offset-scale
    vor_plot_XYs = np.append(plot_XYs, [[x_max + 2.*x_range, y_max + 2.*y_range],
                                        [x_max + 2.*x_range, y_min - 2.*y_range],
                                        [x_min - 2.*x_range, y_max + 2.*y_range],
                                        [x_min - 2.*x_range, y_min - 2.*y_range]],
                             axis=0)
    
    vor = Voronoi(vor_plot_XYs)
    
    # Plot filled tessellation
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    ax1.set_xlabel(r'x (")', fontsize=28)
    ax1.set_ylabel(r'y (")', fontsize=28)
    
    # Uniform color bar
    if (plot_color_lims is not None) and (len(plot_color_lims) == 2):
        color_normalizer = mpl.colors.Normalize(vmin=np.min(plot_color_lims),
                                                vmax=np.max(plot_color_lims))
        custom_color_lims = True
        
        if plot_color_lims[0] > plot_color_lims[1]:
            plot_color_cmap = plot_color_cmap + '_r'
    else:
        color_normalizer = mpl.colors.Normalize(vmin=np.min(plot_quant),
                                                vmax=np.max(plot_quant))
    
    color_cmap = plt.get_cmap(plot_color_cmap)
    
    # Go through each point
    for point_index in range(len(plot_XYs)):
        cur_plot_quant = plot_quant[point_index]
        
        cur_color = color_cmap(color_normalizer(cur_plot_quant))
        
        cur_point_region = vor.point_region[point_index]
        cur_region_vertex_indices = vor.regions[cur_point_region]
        
        cur_vertices_x = np.array([])
        cur_vertices_y = np.array([])
        
        for cur_region_vertex_index in cur_region_vertex_indices:
            if cur_region_vertex_index != -1:
                cur_vertices_x = np.append(cur_vertices_x, (vor.vertices[cur_region_vertex_index])[0])
                cur_vertices_y = np.append(cur_vertices_y, (vor.vertices[cur_region_vertex_index])[1])
        
        cur_vertices = np.stack((cur_vertices_x, cur_vertices_y), axis=-1)
        
        cur_polygon = mpl.patches.Polygon(cur_vertices, color=cur_color)
        ax1.add_artist(cur_polygon)
    
    voronoi_plot_2d(vor, ax=ax1,
                    show_points=False, show_vertices=False,
                    line_width=0.5)
    
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    asec_nums = [-400,-200,0,200,400]
    labels = ['-4','-2','0','2','4']
    ax1.set_xticks(asec_nums)
    ax1.set_xticklabels(labels)
    ax1.set_yticks(asec_nums)
    ax1.set_yticklabels(labels)
    plt.rc('xtick', labelsize=27)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=27)

    #Plot Guide Star (Optional) # Commented out, GS is always on-axis.
    #plt.scatter(guidestar[0], guidestar[1], zorder=2, s=400, edgecolor='k', linewidth=0.5, color='yellow', marker='*')

    if ttstar != None:
        x_vals = (guidestar[0], ttstar[0])
        y_vals = (guidestar[1], ttstar[1])
        plt.plot(x_vals, y_vals, linewidth=6.0, color='k')
        plt.plot(x_vals, y_vals, linewidth=4.5, color='pink')

    #Plot PSF reference stars (Optional)
    if refstars != None:
        psfstars = plt.scatter((refstars[:,3])-512, (refstars[:,4])-512,
            zorder=3, s=350, c=refstars[:,1], edgecolor='k', marker="o", linewidth=2.0,
            cmap=plt.get_cmap(plot_mag_map), vmin=refstars[0,1], vmax=refstars[(len(refstars)-1),1])

    # Color bar
    sc_map = mpl.cm.ScalarMappable(norm=color_normalizer, cmap=color_cmap)
    sc_map.set_array([])
    
    cbar_quant = plt.colorbar(sc_map, ax=ax1, orientation='vertical',
                 label=plot_quant_label)
    cbar_quant.set_label(plot_quant_label, size=28)

    divider = make_axes_locatable(ax1)
    cax = divider.new_vertical(size='5%', pad=0.4)
    top_cbar = fig.add_axes(cax)
    cbar_mag = plt.colorbar(psfstars, cax=cax, orientation='horizontal')
    cbar_mag.set_label('K mag', size=28)
    top_cbar.xaxis.set_ticks_position('top')
    top_cbar.xaxis.set_label_position('top')
    
    # Finish and save out plot
    #fig.tight_layout()
    fig.savefig(plot_out_file_name + '.pdf')
    fig.savefig(plot_out_file_name + '.png', dpi=300)
    plt.close(fig)
