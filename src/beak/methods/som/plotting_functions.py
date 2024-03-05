# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:10:28 2021

Functions for creating plots.
@author: shautala
"""
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import RegularPolyCollection
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def contrasting_text_color(hex_array):
    """Define text color to be black/white to give maximum contrast and visibility

    Args:
        hex_array (array): _description_

    Returns:
        array: black/white text color
    """    
    (r, g, b) = (hex_array[0], hex_array[1], hex_array[2])
    return [0,0,0,1] if 1 - (int(r) * 0.299 + int(g) * 0.587 + int(b) * 0.114)< 0.5 else [1,1,1,1] #Photometric/digital ITU BT.709:: 0.2126 R + 0.7152 G + 0.0722 B    or Digital ITU BT.601 (gives more weight to the R and B components):  0.299 R + 0.587 G + 0.114 B



def plot_hexa(somx,
            somy,
            clusters,
            grid,
            d_matrix,
            annot_ticks=None,
            cluster_tick_labels=None,         
             w=1080,
            dpi=72.,
            title='SOM Hit map',
            colmap='jet',
            ptype='scatter', 
            labelIndex="-2"
            ):
    """Function for plotting hexagonal heatmaps.
        Based on an excellent stackoverflow post by Fernando Ferreira 
        https://stackoverflow.com/questions/23726564/matplotlib-regularpolygon-collection-location-on-the-canvas/23811383#23811383  

    Args:
        somx (int): size of som grid in x
        somy (int): size of som grid in y
        clusters (int): number of clusters
        grid (dict): dictionary holding x, y and center point information if grid type is hexagonal
        d_matrix (_type_): array contaning the distances between each neuron
        annot_ticks (ndarray, optional): annotation ticks. Defaults to None.
        cluster_tick_labels (ndarray, optional): array with tick labels for clusters
        w (int, optional): width of the map in inches. Defaults to 1080.
        dpi (int, optional): dpi. Defaults to 72..
        title (str, optional): title. Defaults to 'SOM Hit map'.
        colmap (str, optional): colormap. Defaults to 'jet'.
        ptype (str, optional): data type. Defaults to 'scatter'.
        labelIndex (str, optional): label index. Defaults to "-2".

    Returns:
        Axes: figure
    """    
    
    discrete_cmap=sns.cubehelix_palette(n_colors=clusters, start=1,rot=4, gamma=1.0, hue=3, light=0.77, dark=0.15, reverse=False, as_cmap=False)
    n_centers = grid['centers']
    x, y = grid['x'], grid['y']
    if(somx<somy):
        xinch = (x * w / y) / dpi
    else:
        xinch=(y * w / y) / dpi
    yinch = (y * w / x) / dpi


    
    fig = plt.figure(figsize=(float(xinch), float(yinch)), dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    
    xpoints = n_centers[:, 0]
    ypoints = n_centers[:, 1]
    #problem block
    ax.scatter(xpoints, ypoints, s=0.0, marker='s')
    ax.axis([min(xpoints)-1., max(xpoints)+1.,
             max(ypoints)+1., min(ypoints)-1.])
    xy_pixels = ax.transData.transform(np.vstack([xpoints, ypoints]).T)
    xpix, ypix = xy_pixels.T
    
    # In matplotlib, 0,0 is the lower left corner, whereas it's usually the
    # upper right for most image software, so we'll flip the y-coords
    width, height = fig.canvas.get_width_height()
    ypix = height - ypix
    ###problem block
    
    if(ptype=='scatter'): #if the data type is csv with gaps
        apothem_x=(xpix[1] - xpix[0]) 
        apothem_y=(ypix[1] - ypix[0]) 
        area_inner_circle = max(abs(apothem_x),abs(apothem_y),4)
        if(area_inner_circle<1):
            apothem=(ypix[1] - ypix[0]) 
            area_inner_circle = abs(apothem) 
        collection_bg = RegularPolyCollection(
            numsides=4,  
            rotation=150,
            sizes=(area_inner_circle,),
            edgecolor="none",
            array= d_matrix,
            cmap = colmap,
            offsets = n_centers,
            transOffset = ax.transData,
        )
        
    else: #regular hexa plot
        apothem = 1.8 * (xpix[1] - xpix[0]) / math.sqrt(3)
        area_inner_circle = math.pi * (apothem ** 2)
        collection_bg = RegularPolyCollection(
            numsides=6, 
            rotation=0,
            sizes=(area_inner_circle,),
            edgecolors = (0, 0, 0, 1),
            array= d_matrix,
            cmap = colmap,
            offsets = n_centers,
            transOffset = ax.transData,
        )
    ax.add_collection(collection_bg, autolim=True) 
    ax.axis('off')
    ax.autoscale_view()
    divider = make_axes_locatable(ax)
    
    if(colmap=='jet'):
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(collection_bg, cax=cax,ticklocation='right', aspect=10,spacing='proportional')
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(axis='y', direction='out', pad=30)
    elif(colmap=='binary'):
        if(x<y):
            cax = divider.append_axes("right", size="5%", pad=0.05)
            mpl.rcParams.update({'font.size': int((y/x)*32)})  
        else:
            cax = fig.add_axes([.98,.124,.04,.754])
        cbar=plt.colorbar(collection_bg, cax=cax,ticklocation='right', aspect=10,spacing='proportional')
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(axis='y', direction='out', pad=30)
    else:
        if(x>y and clusters>15):
            mpl.rcParams.update({'font.size': int((y/x)*32)})  
        colmap2=colmap
        if(str(labelIndex)!="-2"):
            ticks_flattened=annot_ticks.flatten()
            for i in range(0, len(n_centers)):
                ax.annotate(ticks_flattened[i], (n_centers[i][0], n_centers[i][1]),color=contrasting_text_color(discrete_cmap[int(d_matrix[i])]),ha='center', va='center', fontsize=max(39-(somy/2)-(4*somx/somy),10))
        bounds = np.linspace(0, clusters, clusters+1)
        bounds2 = np.linspace(0.5, clusters-0.5, clusters)#clusters+0.5 in case of som cell selection?
        norm = mpl.colors.BoundaryNorm(bounds, colmap2.N)
        if(ptype=='scatter'):
            cb_ax = divider.append_axes("right", size="5%", pad=0.05)
        else:
            if(x<y):
                cb_ax = divider.append_axes("right", size="5%", pad=0.05)
            else:
                cb_ax = fig.add_axes([.98,.124,.04,.754])
                   
        cbar=mpl.colorbar.ColorbarBase(cb_ax, cmap=colmap2, norm=norm,spacing='uniform',ticklocation='right', ticks=bounds2, boundaries=bounds, format='%1i')
        cbar.ax.invert_yaxis()   
        cbar.ax.tick_params(axis='y', direction='out', pad=30)
        ticks_reverse=cluster_tick_labels.copy()
        ticks_reverse.reverse()
        cbar.set_ticklabels(ticks_reverse)
    plt.gca().invert_yaxis()
    ax.set_title(title)

    return ax
