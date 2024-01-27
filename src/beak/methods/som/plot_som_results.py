# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:54:21 2019

@author: Sakari Hautala, Joonas Haikonen

Python script to visualize SOM calculation results & original data. Draws heatmaps based on som and geospace results, and boxplots of data distribution per cluster 

Inputs:
1) Somspace results file
2) Som x dimension
3) Som y dimension
4) Geospace results file
5) Input file used for som calculation 
6) Index of easting column 
7) Index of northing column
8) Index of label column
8) Output folder
9) Grid type (rectangular or hexagonal)
10) Redraw (boolean) - whether to calculate all plots or only those that deal with clustering 
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker # HPacker,
from matplotlib.ticker import FormatStrFormatter
import math
from matplotlib.lines import Line2D
from .plotting_functions import plot_hexa
import time
import warnings

"""
Run plotting scripts
"""    
def run_plotting_script(argsP):
    print("Setup figures")
    start_time = time.time()

    [geo_data, geo_headers, 
     som_data, som_table, som_headers, som_dict,
     grid, grid_type, annot_ticks, annot_strings, 
     outgeofile, 
     clusters, cluster_ticks, cluster_tick_labels,
     discrete_cmap, discrete_cmap_2, 
     labelIndex
     ] = basic_setup(
        argsP.outsomfile, argsP.som_x, argsP.som_y, 
        argsP.input_file, argsP.dir, 
        argsP.grid_type, argsP.redraw, 
        argsP.dataType, argsP.noDataValue, argsP.outgeofile)

    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")

    if argsP.outgeofile is not None: #if spatial, draw geo plots
        print("Plot geo space results")
        start_time = time.time()
        if(argsP.dataType=='scatter'):
            if(clusters>1):
                plot_geospace_clusters_scatter(geo_data, discrete_cmap_2, argsP.dir)
            if(argsP.redraw!="false"):
                plot_geospace_results_scatter(geo_data, geo_headers, som_data, argsP.dir)
        else:
            if(clusters>1):#if clusters
                plot_geospace_clusters_grid(geo_data, discrete_cmap, clusters,cluster_ticks, cluster_tick_labels, argsP.dir)
            if(argsP.redraw!="false"):
                plot_geospace_results_grid(geo_data, geo_headers, som_data, argsP.dir, argsP.noDataValue)

        end_time = time.time()
        print(f"    Execution time: {end_time - start_time} seconds")

    if(clusters>1): 
        print("Plot Cluster result SOM space")
        start_time = time.time()
        #draw som cluster plot if there is more than 1 cluster
        draw_som_clusters(som_data, som_table, annot_ticks, som_headers, discrete_cmap, discrete_cmap_2, argsP.dir, grid_type, clusters, cluster_ticks, cluster_tick_labels, labelIndex, annot_strings)
        # Load cluster dictionary
        loaded_cluster_list = load_cluster_dictionary(argsP.dir)
        # Plot and save the Davies-Bouldin Index vs Number of Clusters
        print("    Plot Davies Bouldin index")
        plot_davies_bouldin(loaded_cluster_list, argsP.dir)
        print("    Plot cluster hit count")
        plot_cluster_hit_count(argsP.dir+"/cluster_hit_count.txt", argsP.dir)

        end_time = time.time()
        print(f"    Execution time: {end_time - start_time} seconds")

    print("Plot SOM space results")
    start_time = time.time()

    draw_umatrix(som_data, som_table, grid, grid_type, annot_ticks, som_headers, argsP.dir)
    draw_number_of_hits(som_dict,som_data,argsP.som_x,argsP.som_y,clusters,grid,cluster_tick_labels,grid_type,argsP.dir)
    #in case the function was called for redrawing after selecting a different clustering result. so that we can skip stuff we don't have to redraw to speed things up. CURRENTLY NOT IN USE, ALWAYS TRUE.
    if(argsP.redraw!="false"):
        draw_som_results(som_data, som_table,grid, grid_type, annot_ticks, som_headers, argsP.dir)

    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")
    
    #print("SomSpace plots finshed")

    if(som_dict['clusters'] is not None):
        print("Plot Boxplots")
        start_time = time.time()

        draw_boxplots(som_dict,som_data,som_headers,discrete_cmap, cluster_tick_labels, argsP.dir)
        #print("Boxplots finished")
        end_time = time.time()
        print(f"    Execution time: {end_time - start_time} seconds")



def basic_setup(outsomfile, som_x, som_y, input_file, working_dir, grid_type, redraw, dataType, noDataValue, aOutgeofile):

    """Load input parameters & do basic setup"""
    """Initialize variables"""
 
    somx=int(som_x)        
    somy=int(som_y)
    outgeofile=None          
    geo_data=None
    eastingIndex=None
    northingIndex=None
    labelIndex=None 

    if aOutgeofile is not None:
        outgeofile=aOutgeofile    

    with open(working_dir+"/som.dictionary", 'rb') as som_dictionary_file:
         som_dict = pickle.load(som_dictionary_file)

    time_0 = time.time()

    som_data = np.genfromtxt(outsomfile,skip_header=(1), delimiter=' ')
    som_headers=pd.read_csv(outsomfile, delimiter=' ', header=None).iloc[0] 

    time_1 = time.time()
    print(f"        Read som data execution time: {time_1 - time_0} seconds")

    if outgeofile is not None:
        #geo_data=np.genfromtxt(outgeofile, skip_header=(1), delimiter=' ')
        #time_A = time.time()
        #print(f"        Read geo data (genfromtxt):                 {time_A - time_1} seconds")
        geo_data = pd.read_csv(outgeofile, skiprows=1, delimiter=' ').values
        time_A = time.time()
        #print(f"        Read geo data (read_csv):                 {time_A - time_1} seconds")
        #if spatial, draw geo plots
        geofile = open(outgeofile, "r")       
        header_line = geofile.readline() 
        geo_headers=['#']
        geo_headers = geo_headers +header_line.split(" ")

        time_2 = time.time()
        print(f"        Read geo data execution time: {time_2 - time_1} seconds")

    som_table=np.zeros((somx,somy))#empty somx*somy sized table for som plots

    #Generate colormaps and ticks for clustering
    clusters=int(max(som_data[:,len(som_data[0])-2])+1)
    discrete_cmap=sns.cubehelix_palette(n_colors=clusters, start=1,rot=4, gamma=1.0, hue=3, light=0.77, dark=0.15, reverse=False, as_cmap=False)
    discrete_cmap_2=sns.cubehelix_palette(n_colors=clusters, start=1,rot=4, gamma=1.0, hue=3, light=0.77, dark=0.15, reverse=False, as_cmap=True)

    # Define the maximum number of labels
    max_labels = 10

    # Calculate the label interval dynamically based on max_labels
    label_interval = max(1, (clusters + max_labels - 1) // max_labels)
    cluster_ticks = np.arange(0, clusters + 1, label_interval)
    
    # Create cluster_tick_labels with the dynamically calculated interval
    #cluster_tick_labels = [f"{i} {cluster_hit_count[i]}" for i in range(0, clusters+1, label_interval)]
    cluster_tick_labels = cluster_ticks

    palette=sns.cubehelix_palette(n_colors=clusters, start=1,rot=4, gamma=1.0, hue=3, light=0.77, dark=0.15, reverse=False, as_cmap=False)
    formatted_palette = [f'rgb({int(j*255)},{int(j*255)},{int(j*255)})' for i in palette for j in i] #[]

    #Format palette into colorscale. for example 10 clusters: (0,0.1,rgb_val), (0.1,0.2 rgb_val_2),...... ,(0.9,1,rgb_val_x) ((not always distance of 0.1)) so each cluster is assigned a specific color.
    clusterColorscale = [[i/clusters, formatted_palette[i]] for i in range(len(formatted_palette))] 
    clusterColorscale = [item for sublist in zip(clusterColorscale[:-1], clusterColorscale[1:]) for item in sublist]


    if(grid_type.lower()=="hexagonal"): #if grid shape is hexagonal, initialize corresponding variables for hexa plots
        x=somx
        y=somy
        centers = [(i + 1 + (0 if j % 2 == 0 else 0.5), (j + 1) * math.sqrt(3) / 2) for i in range(x) for j in range(y)] #is this correcly replacing the loops below?
        grid={'centers':np.array(centers), 
            'x':np.array([float(x)]),
            'y':np.array([float(y)])}
    else:
        grid=None


    # Read data using space delimiter if input file is GeoTIFF
    data_file = outgeofile if outgeofile is not None else input_file
    with open(data_file, encoding='utf-8-sig') as fh:
        header_line = fh.readline()

    colnames = header_line.split() if outgeofile is not None else header_line.split("\t")

    labelIndex = colnames.index('label') if 'label' in colnames else None

    # Process label data
    annot_ticks = np.empty([somx, somy], dtype='<U32')
    annot_ticks.fill("")

    annot_strings = {}
    annot_data = []

    if 'label' in colnames:
        label_index = colnames.index('label')
        data = np.loadtxt(data_file, dtype='str', delimiter=' ' if outgeofile is not None else '\t',
                          skiprows=0 if outgeofile is not None else 3)

        for i in range(len(data)):
            if data[i][label_index] not in ['', "nan", "NA", "NULL", "Null", "NoData", noDataValue]:
                tick = annot_ticks[som_dict['bmus'][i][0]][som_dict['bmus'][i][1]]
                counter = len(annot_strings) + 1

                if tick == '':
                    annot_ticks[som_dict['bmus'][i][0]][som_dict['bmus'][i][1]] = str(counter)
                    annot_strings[str(counter)] = [data[i][label_index]]
                    annot_data.append([f"{counter}: {data[i][label_index]}", f"{som_dict['bmus'][i][0]}{som_dict['bmus'][i][1]}",
                                       f"{geo_data[i][0]}, {geo_data[i][1]}" if outgeofile is not None else None])
                else:
                    annot_strings[tick].append(data[i][label_index])
                    annot_data.append([f"{tick}: {data[i][label_index]}", f"{som_dict['bmus'][i][0]}{som_dict['bmus'][i][1]}",
                                       f"{geo_data[i][0]}, {geo_data[i][1]}" if outgeofile is not None else None])

        # Merge duplicates within a labeling group
        for i, j in itertools.combinations(range(1, counter + 1), 2):
            if annot_strings.get(str(i)) == annot_strings.get(str(j)):
                annot_strings.pop(str(j), None)
                for a, b in itertools.product(range(len(annot_ticks)), range(len(annot_ticks[0]))):
                    if annot_ticks[a][b] == str(j):
                        annot_ticks[a][b] = str(i)

        # Remove gaps in index numbers
        counter = 0
        for i in range(1, counter + 1):
            if str(i) in annot_strings:
                counter += 1
                annot_strings[str(counter)] = annot_strings.pop(str(i))
                for a, b in itertools.product(range(len(annot_ticks)), range(len(annot_ticks[0]))):
                    if annot_ticks[a][b] == str(i):
                        annot_ticks[a][b] = str(counter)

        # Format ticks
        for i in range(1, len(annot_strings) + 1):
            annot_strings[str(i)] = f"{i}: {','.join(annot_strings[str(i)])}"

    #return {'var1': var1, 'var2':var2}
    return geo_data, geo_headers, som_data, som_table, som_headers, som_dict, grid, grid_type, annot_ticks, annot_strings, outgeofile, clusters, cluster_ticks, cluster_tick_labels, discrete_cmap, discrete_cmap_2, labelIndex
    #return {'geo_data': geo_data, 'geo_headers': geo_headers, 'som_data': som_data, 'som_table': som_table, 'som_headers': som_headers, 'som_dict': som_dict, 'grid': grid, 'annot_ticks': annot_ticks, 'outgeofile': outgeofile, 'clusters': clusters}


"""
Plot geospace plots & q-error if type is grid
"""
def plot_geospace_results_grid(geo_data, geo_headers, som_data, working_dir, noDataValue):
    mpl.rcParams.update({'font.size': 14})

    for i in range(0, len(som_data[0])-4): 
        print(f"    geospace plot no. {i+2} from {len(som_data[0])-3}", end='\r')  
        x=geo_data[:,0]      
        y=geo_data[:,1]
        z=geo_data[:,(5+i)]

        # Replace noDataValues with np.nan
        z[z == noDataValue] = np.nan

        df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
        df.columns = ['X_value','Y_value','Z_value']
        df['Z_value'] = pd.to_numeric(df['Z_value'])
        pivotted= df.pivot(index='Y_value', columns='X_value', values='Z_value')
        
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        ax=sns.heatmap(pivotted,cmap='jet', square=True, linewidths=0, xticklabels="auto", yticklabels="auto", mask=np.isnan(pivotted))

        # Set tick labels
        fmt = '{:0.0f}'
        xticklabels = []
        for item in ax.get_xticklabels():
            item.set_text(fmt.format(float(item.get_text())))
            xticklabels += [item]
        yticklabels = []
        for item in ax.get_yticklabels():
            item.set_text(fmt.format(float(item.get_text())))
            yticklabels += [item]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)

        every_nth = round((len(ax.xaxis.get_ticklabels()))/2)
        if(every_nth==0):#for data sets with only 1 x coordinate
            every_nth=1
        every_nth_y = round((len(ax.yaxis.get_ticklabels()))/2)
        if(every_nth_y==0):#for data sets with only 1 x coordinate
            every_nth_y=1
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_nth_y != 0:
                label.set_visible(False)
        ax.xaxis.get_ticklabels()[-1].set_visible(True)
        ax.yaxis.get_ticklabels()[-1].set_visible(True)
        plt.yticks(rotation=90,ha='right',va='bottom')
        plt.xticks(rotation=0,ha='left')
        ax.invert_yaxis()
        ax.set_title(geo_headers[5+i+1])
        plt.tight_layout()
        ax.figure.savefig(working_dir+'/geoplot_'+str(i+2)+'.png', dpi=300)
        plt.clf()
        plt.cla()
        plt.close()
    print()    
        
    #q_error:
    print(f"    q-error plot")
    x=geo_data[:,0]
    y=geo_data[:,1]
    z=geo_data[:,(len(som_data[0])-5)*2 +5] 
    df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
    df.columns = ['X_value','Y_value','Z_value']
    df['Z_value'] = pd.to_numeric(df['Z_value'])
    pivotted= df.pivot(index='Y_value', columns='X_value', values='Z_value')
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    ax=sns.heatmap(pivotted,cmap='jet', square=True, linewidths=0, xticklabels="auto", yticklabels="auto")
    
    # Set tick labels
    fmt = '{:0.0f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    every_nth = round((len(ax.xaxis.get_ticklabels()))/2)
    if(every_nth==0):#for data sets with only 1 x coordinate
        every_nth=1
    every_nth_y = round((len(ax.yaxis.get_ticklabels()))/2)
    if(every_nth_y==0):#for data sets with only 1 x coordinate
        every_nth_y=1
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth_y != 0:
            label.set_visible(False)
    ax.xaxis.get_ticklabels()[-1].set_visible(True)
    ax.yaxis.get_ticklabels()[-1].set_visible(True)
    plt.yticks(rotation=90, ha='right',va='bottom')
    plt.xticks(rotation=0, ha='left')
    ax.invert_yaxis()
    ax.set_title(geo_headers[(len(som_data[0])-5)*2 +6])#(geo_headers[len(som_data[0])+i+1])#tähän sama 5     
    plt.tight_layout()
    ax.figure.savefig(working_dir+'/geoplot_'+str(i+2)+'.png', dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    
    

"""
Plot geospace plots & q-error if type is scatter
"""
def plot_geospace_results_scatter(geo_data, geo_headers, som_data, working_dir):

    centers=[]     
    for i in range(0, len(geo_data)):	
        centers.append([geo_data[i][0],geo_data[i][1]])
    grid={'centers':np.array(centers), 
          'x':np.array([len(geo_data)]),
          'y':np.array([len(geo_data)])}

    for i in range(0, len(som_data[0])-4):  
        print(f"    geospace plot no. {i+2} from {len(som_data[0])-3}", end='\r')
        z=geo_data[:,(5+i)]   
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        mpl.rcParams.update({'font.size': 30})
        ax = plot_hexa(somx,somy,clusters,grid,z,cluster_tick_labels=cluster_tick_labels, title=geo_headers[5+i+1], ptype='scatter')           
        plt.yticks(rotation=90, ha='right', va='bottom')
        plt.xticks(rotation=0, ha='left')
        ax.invert_yaxis()
        plt.tight_layout()
        ax.figure.savefig(working_dir+'/geoplot_'+str(i+2)+'.png', dpi=300)
        plt.clf()
        plt.cla()
        plt.close()
        mpl.rcParams.update({'font.size': 12})  
    print()
        
    #draw q_error:
    print(f"    q-error plot")
    z=geo_data[:,(len(som_data[0])-5)*2 +5]   
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    mpl.rcParams.update({'font.size': 30})
    ax = plot_hexa(somx,somy,clusters,grid,z,cluster_tick_labels=cluster_tick_labels, title=geo_headers[(len(som_data[0])-5)*2 +6], ptype='scatter')           
    plt.yticks(rotation=90, ha='right', va='bottom')
    plt.xticks(rotation=0, ha='left')
    ax.invert_yaxis()
    plt.tight_layout()
    ax.figure.savefig(working_dir+'/geoplot_'+str(len(som_data[0])-2)+'.png', dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    mpl.rcParams.update({'font.size': 12})


"""
Draw Som result plots
"""
def draw_som_results(som_data, som_table,grid, grid_type, annot_ticks, som_headers,working_dir):
    for j in range(2,len(som_data[0])-3):
        print(f"    somspace plot no. {j-1} from {len(som_data[0])-5}", end='\r')
        if(grid_type.lower()=="rectangular"):
            for i in range(0,len(som_data)): 
                som_table[int(som_data[i][0])][int(som_data[i][1])]=som_data[i][j] #som_table: somx*somy size
            ax = sns.heatmap(som_table.transpose(), cmap="jet", linewidth=0)   
            ax.set_title(som_headers[j])    
        else:#grid type=="hexagonal":
            hits=som_data[:,j]
            mpl.rcParams.update({'font.size': 30})
            ax = plot_hexa(somx,somy,clusters,grid, hits, annot_ticks,cluster_tick_labels,title=som_headers[j+1], ptype='grid')   
            mpl.rcParams.update({'font.size': 32})           
            ax.set_title(som_headers[j]) 
            mpl.rcParams.update({'font.size': 32})  
        ax.figure.savefig(working_dir+'/somplot_' +str(j-1)+'.png',bbox_inches='tight')#Creating the folder is done in C# side of things.    
        plt.clf()
        plt.cla()
        plt.close()  
        mpl.rcParams.update({'font.size': 12})
    print()
        
"""
Draw U-matrix plot
"""
def draw_umatrix(som_data, som_table,grid, grid_type, annot_ticks, som_headers,working_dir):
    for j in range(len(som_data[0])-3,len(som_data[0])-2):
        if(grid_type.lower()=="rectangular"):
            for i in range(0,len(som_data)): 
                som_table[int(som_data[i][0])][int(som_data[i][1])]=som_data[i][j] #som_table: somx*somy size
            ax = sns.heatmap(som_table.transpose(), cmap="jet", linewidth=0)   
            ax.set_title(som_headers[j])    
        else:#grid type=="hexagonal":
            hits=som_data[:,j]
            mpl.rcParams.update({'font.size': 30})
            ax = plot_hexa(somx,somy,clusters,grid, hits, annot_ticks,cluster_tick_labels,title=som_headers[j+1], ptype='grid')   #j+1 to 
            mpl.rcParams.update({'font.size': 32})           
            ax.set_title(som_headers[j]) 
            mpl.rcParams.update({'font.size': 32})  
        ax.figure.savefig(working_dir+'/somplot_' +str(j-1)+'.png',bbox_inches='tight')#Creating the folder is done in C# side of things.    
        plt.clf()
        plt.cla()
        plt.close()  
        mpl.rcParams.update({'font.size': 12})

"""
Draw Som Cluster plot
"""
def draw_som_clusters(som_data, som_table, annot_ticks, som_headers,discrete_cmap,discrete_cmap_2,working_dir,grid_type,clusters,cluster_ticks,cluster_tick_labels, labelIndex, annot_strings):    
    if(grid_type.lower()=="rectangular"):
        mpl.rcParams.update({'font.size': 14})  
        for i in range(0,len(som_data)): 
            som_table[int(som_data[i][0])][int(som_data[i][1])]=som_data[i][len(som_data[0])-2]          
        ax = sns.heatmap(som_table.transpose(), cmap=discrete_cmap, linewidth=0, vmin=-0.5, vmax=clusters-0.5, center=clusters/2-0.5, cbar_kws=dict(ticks=cluster_ticks), annot=annot_ticks.transpose(), fmt = '')
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticklabels(cluster_tick_labels)
        ax.set_title(som_headers[len(som_headers)-2])
    else:#grid type=="hexagonal":
        hits=som_data[:,len(som_data[0])-2]   
        mpl.rcParams.update({'font.size': 30})  
        ax = plot_hexa(somx,somy,clusters,grid,hits,annot_ticks,cluster_tick_labels,colmap=discrete_cmap_2, ptype='grid',labelIndex=labelIndex)
        mpl.rcParams.update({'font.size': 32})  
        ax.set_title(som_headers[len(som_headers)-2])
        mpl.rcParams.update({'font.size': 30})  
    ax.figure.savefig(working_dir+'/somplot_' + str(len(som_data[0])-3) + '.png',bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    if(labelIndex!=None):
        mpl.rcParams.update({'font.size': 12})  
        fig = plt.figure(figsize= [6.4, 4.8])
        ax1 = fig.add_subplot(211) # Creates another plot image and turns visibility of the axes off
        ax1.axis('off')
        children=[]
        for text in annot_strings:
            children.append(TextArea(annot_strings[text], textprops=dict(color="red")))
        box = VPacker(children=children, align="left", pad=5, sep=5)

        # anchored_box creates the text box outside of the plot
        if(grid_type.lower()=="rectangular"):
            location=3
            anchored_box = AnchoredOffsetbox(loc=location,
                                            child=box, pad=0.,
                                            frameon=True,
                                            bbox_to_anchor=(0.01, 0.01),
                                            bbox_transform=ax.transAxes,
                                            borderpad=0.,                                       
                                            )
        else:
            location=3
            anchored_box = AnchoredOffsetbox(loc=location,
                                            child=box,
                                            borderpad=0.
                                            )
        
        ax1.add_artist(anchored_box)
        ax1.figure.savefig(working_dir+'/somplot_' + str(len(som_data[0])-1) + '.png',bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
        df = pd.DataFrame(annot_data)
        if outgeofile is not None: 
            headers = ["label", "som", "datapoint"]
        else:
            headers = ["label", "som"]
        df.to_csv(working_dir+'/labels_flat.csv', index=False, header=headers)   #so in addition to this there should be a list thats written out in the format of current label legend?
        #list_grouped=list(annot_strings.items())
        array_grouped=np.array(list(annot_strings.items()))#dict to list and list to np array
        for i in range(0,len(array_grouped)):
            array_grouped[i][1]= array_grouped[i][1][(array_grouped[i][1].find(":")+1):len(array_grouped[i][1])]      #  ": "+ ','.join(annot_strings[str(i)])
        np.savetxt(working_dir+'/labels_grouped.csv', array_grouped, delimiter=',', fmt='%s')
        #df_grouped.to_csv(working_dir+'/labels_grouped.csv', index=False, header=headers)

"""
Plot geospace clusters, if there is more than 1 cluster and input type is grid
"""
def plot_geospace_clusters_grid(geo_data,discrete_cmap,clusters,cluster_ticks,cluster_tick_labels,working_dir):
    #global geo_data
    x=geo_data[:,0]
    y=geo_data[:,1]
    z=geo_data[:,(4)]    
    df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
    df.columns = ['X_value','Y_value','Z_value']
    df['Z_value'] = pd.to_numeric(df['Z_value'])
    pivotted= df.pivot(index='Y_value', columns='X_value', values='Z_value')
    mpl.rcParams.update({'font.size': 14})
    ax=sns.heatmap(pivotted,cmap=discrete_cmap, vmin = -0.5, vmax = clusters - 0.5, square=True, linewidths=0, xticklabels="auto", yticklabels="auto", cbar_kws=dict(ticks=cluster_ticks))

    # Set tick labels to integers:
    fmt = '{:0.0f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    every_nth = round((len(ax.xaxis.get_ticklabels()))/2)
    if(every_nth==0):# for data sets with only 1 x coordinate
        every_nth=1
    every_nth_y = round((len(ax.yaxis.get_ticklabels()))/2)
    if(every_nth_y==0):#for data sets with only 1 y coordinate
        every_nth_y=1
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth_y != 0:
            label.set_visible(False)
    ax.xaxis.get_ticklabels()[-1].set_visible(True)
    ax.yaxis.get_ticklabels()[-1].set_visible(True)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(cluster_tick_labels)
    plt.yticks(rotation=90)#, ha='right', va='bottom')
    #plt.yticks(ha='right')
    #plt.yticks(va='bottom')
    plt.xticks(rotation=0)
    plt.xticks(ha='left')
    ax.invert_yaxis()
    ax.set_title('cluster')
    plt.tight_layout()
    ax.figure.savefig(working_dir+'/geoplot_'+str(1)+'.png', dpi=300)
    plt.clf()
    plt.cla()
    plt.close()       
    
    
"""
Plot geospace clusters if input type is scatter
"""
def plot_geospace_clusters_scatter(geo_data,discrete_cmap_2,working_dir):
    #global geo_data
    z=geo_data[:,(4)]  
    centers = geo_data[:, :2]  # Directly create a NumPy array

    #centers=[]     
    #for i in range(0, len(geo_data)):	
    #    centers.append([geo_data[i][0],geo_data[i][1]])

    #grid={'centers':np.array(centers), 
    #      'x':np.array([len(geo_data)]),
    #      'y':np.array([len(geo_data)])}  
    
    grid = {'centers': centers,
            'x': np.array([len(geo_data)]),
            'y': np.array([len(geo_data)])}

    mpl.rcParams.update({'font.size': 30})
    ax = plot_hexa(somx,somy,clusters,grid,z,cluster_tick_labels=cluster_tick_labels, title="clusters",colmap=discrete_cmap_2, ptype='scatter')       

    ax.invert_yaxis()
    ax.set_title('cluster')
    plt.tight_layout()
    ax.figure.savefig(working_dir+'/geoplot_'+str(1)+'.png', dpi=300)

    plt.clf()
    plt.cla()
    plt.close()   
    mpl.rcParams.update({'font.size': 12})  


    
"""
Plot boxplots using som data.
"""

def draw_boxplots(som_dict,som_data,som_headers,discrete_cmap,cluster_tick_labels,working_dir):
    
    mpl.rcParams.update({'font.size': 12})  
    cluster_col=[]
    
    for i in range(0,len(som_data)): 
        cluster_col.append(som_data[i][len(som_data[0])-2])
    cluster_nparray=np.asarray(cluster_col)   
    clusters_unique=np.unique(cluster_nparray)
    for k in range(len(discrete_cmap)-1,-1,-1):
        if(k not in clusters_unique):
            discrete_cmap.pop(k)  
    
    for i in range(2,len(som_data[0])-3): 
        print(f"    boxplot no. {i-1} from {len(som_data[0])-3-2}", end='\r')
        z=som_data[:,i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ax=sns.boxplot(x=cluster_nparray.astype(float), y=z.astype(float), hue=cluster_nparray.astype(float) ,dodge=False, palette=discrete_cmap)       
        ax.set_title(som_headers[i])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f')) 
        ax.legend_.remove()
        #custom_lines=[]#dummy handle
        #for j in range(0,len(cluster_tick_labels)):
        #    custom_lines.append(Line2D([0], [0], color='blue', lw=4))#only working way I found to make legened properly without handles, was to pass custom width 0 handles. 
        #ax.legend(custom_lines,cluster_tick_labels,bbox_to_anchor=(1.05, 1),loc=0,handlelength=0,fontsize=8,handletextpad=0 ,borderaxespad=0.)   
        plt.tight_layout()           
        ax.figure.savefig(working_dir+'/boxplot_' +str(i-1)+'.png')   
        plt.clf()
        plt.cla()
        plt.close()  
    print()      
        

"""
Draw number of hits
"""
def draw_number_of_hits(som_dict,som_data,somx,somy,clusters,grid,cluster_tick_labels,grid_type,working_dir):
    mpl.rcParams.update({'font.size': 12}) 
    hits=np.zeros((somx,somy))   
    for i in range(0, len(som_dict['bmus'])):
        x=int(som_dict['bmus'][i][0])
        y=int(som_dict['bmus'][i][1])
        hits[x][y]+=1
    hits=np.transpose(hits)
    if(grid_type=='rectangular'):
        ax = sns.heatmap(hits, cmap="binary", linewidth=0)   
        ax.set_title("Number of hits per SOM cell")
    else: #if grid type is hexagonal
        mpl.rcParams.update({'font.size': 30})
        ax = plot_hexa(somx,somy,clusters,grid,hits.flatten(order='F'),annot_ticks,cluster_tick_labels,  colmap="binary", ptype='grid')    
        mpl.rcParams.update({'font.size': 32})  
        if(somy/somx>1.5):
            mpl.rcParams.update({'font.size': int(48/(somy/somx))})  #scale header font down if plot is narrow (i.e. x<y). This was a problem only in this, because the title is so long compared to the others
        #ax.set_title("Number of hits per SOM cell")
 
        mpl.rcParams.update({'font.size': 30})  
    ax.figure.savefig(working_dir+'/somplot_' +str(len(som_data[0])-2)+'.png',bbox_inches='tight')
    mpl.rcParams.update({'font.size': 12})
    plt.clf()
    plt.cla()
    plt.close()
 

"""
Draw Davies-Bouldin Index
"""
def load_cluster_dictionary(file_path):
    """
    Load the clustering dictionary from the specified file path.

    :Param: file_path (str): The path to the 'cluster.dictionary' file.
    :Returns: cluster_list (list): List containing clustering information.
    """
    with open(file_path+"/cluster.dictionary", 'rb') as cluster_dictionary_file:
        cluster_list = pickle.load(cluster_dictionary_file)
    return cluster_list

def plot_davies_bouldin(cluster_list, output_path):
    """
    Plot the number of clusters against the Davies-Bouldin Index and save the plot to a file.
    
    :Params: cluster_list (list): List containing clustering information.
    :Params: output_path (str): The path to save the plot as a PNG file.
    """
    cluster_numbers = [entry['n_clusters'] for entry in cluster_list]
    davies_bouldin_scores = [entry['db_score'] for entry in cluster_list]

    fig, ax = plt.subplots()
    ax.plot(cluster_numbers, davies_bouldin_scores, marker='o')
    ax.set_title('Number of Clusters vs Davies-Bouldin Index')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.grid(True)
    fig.savefig(output_path+"/db_score.png")
    plt.clf()
    plt.cla()
    plt.close()


"""
Draw cluster_hit_count
"""
def plot_cluster_hit_count(txtFile, output_path):
    # Load the cluster_hit_count.txt file
    data = np.genfromtxt(txtFile, delimiter='\t', names=True)

    fig, ax = plt.subplots()
    ax.bar(data['ClusterNumber'], data['HitCount'])
    ax.set_xlabel('Cluster Number')
    ax.set_ylabel('Hit Count')
    ax.set_title('Cluster Hit Count')
    ax.grid(True)
    #plt.show()
    fig.savefig(output_path+"/cluster_hit_count.png")
    plt.clf()
    plt.cla()
    plt.close()