# quant_map_vis.py
# ---
# Script to visualize PSF quantity (e.g. Strehl) on a map
# Authors: Abhimat Gautam

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree

def quant_map_vis(plot_quant, plot_x, plot_y,
                  plot_type='Voronoi',
                  plot_quant_label='',
                  plot_out_file_name='quant_map'):
    # Set up Voronoi plot, and bounds
    x_max = np.max(plot_x)
    x_min = np.min(plot_x)
    x_range = x_max - x_min
    
    y_max = np.max(plot_y)
    y_min = np.min(plot_y)
    y_range = y_max - y_min
    
    plot_XYs = np.stack((plot_x, plot_y), axis=-1)
    vor_plot_XYs = np.append(plot_XYs, [[x_max + 2.*x_range, y_max + 2.*y_range],
                                        [x_max + 2.*x_range, y_min - 2.*y_range],
                                        [x_min - 2.*x_range, y_max + 2.*y_range],
                                        [x_min - 2.*x_range, y_min - 2.*y_range]],
                             axis=0)
    
    vor = Voronoi(vor_plot_XYs)
    
    # Plot filled tessellation
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(1,1,1)
    
    ax1.set_xlabel(r"x")
    ax1.set_ylabel(r"y")
    
    # Uniform color bar
    color_normalizer = mpl.colors.Normalize(vmin=np.min(plot_quant),
                                            vmax=np.max(plot_quant))
    color_cmap = plt.get_cmap('plasma')
    
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
    
    # Color bar
    sc_map = mpl.cm.ScalarMappable(norm=color_normalizer, cmap=color_cmap)
    sc_map.set_array([])
    
    plt.colorbar(sc_map, ax=ax1, orientation='vertical',
                 label=plot_quant_label)
    
    
    # Finish and save out plot
    fig.tight_layout()
    fig.savefig(plot_out_file_name + '.pdf')
    fig.savefig(plot_out_file_name + '.png', dpi=200)
    plt.close(fig)
