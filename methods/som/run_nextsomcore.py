# -*- coding: utf-8 -*-
"""
Script for performing SOM training and saving results.


First 7 parameters (script name, input data, output geo file, output som file, som x, som y, epochs) are required.
Parameters are passed in the following format: '--name=value'. The order of the parameters doesn't matter.
If input contains --xmlfile parameter, other command line parameters will be ignored.
If Initial codebook parameter is given, Initialization parameter is ignored, and the given som x and som y dimensions must match those of the existing codebook.
@author: Janne Kallunki, Sakari Hautala
"""
#import argparse
#import xml.etree.ElementTree as ET
from nextsomcore.nextsomcore import NxtSomCore
import pickle

#import somoclu


#------------------------------------------------------------------------------------------
#--- specify parameter for SOM
#------------------------------------------------------------------------------------------
input_file="/methods/methods/som/nextsomcore/data/input/SOM.lrn"

output_folder="/methods/methods/som/nextsomcore/data/output"         # Folder to save som dictionary and cluster dictionary
output_file_somspace="/methods/methods/som/nextsomcore/data/output/somspace.txt"   
outgeofile="/methods/methods/som/nextsomcore/data/output/geospace.txt"   


# If input data is geotiff: list geotiff files, separated by "," ["name1.tiff","name2.tiff"]
#input_list_text=[]
#input_file= ",".join(input_list_text)
        
som_x=10                # X dimension of generated SOM
som_y=10                # Y dimension of generated SOM
epochs=10               # Number of epochs to run

maptype='toroid'            # Type of SOM (sheet, toroid)
initialcodebook=None        # File path of initial codebook, 2D numpy.array of float32.
neighborhood='gaussian'     # Shape of the neighborhood function. gaussian or bubble
std_coeff=0.5               # Coefficient in the Gaussian neighborhood function
radius0=0                   # Initial size of the neighborhood
radiusN=1                   # Final size of the neighborhood
radiuscooling='linear'      # Function that defines the decrease in the neighborhood size as the training proceeds (linear, exponential)
scalecooling='linear'       # Function that defines the decrease in the learning scale as the training proceeds (linear, exponential)
scale0=0.1                  # Initial learning rate
scaleN=0.01                 # Final learning rate
initialization='random'     # Type of SOM initialization (random, pca)
gridtype='rectangular'      # Type of SOM grid (hexagonal, rectangular)
#xmlfile="none"              # SOM inputs as an xml file

kmeans="true"           # Run k-means clustering
kmeans_init=5           # Number of initializations
kmeans_min=2            # Minimum number of k-mean clusters
kmeans_max=25           # Maximum number of k-mean clusters

#output_file_geospace=None   # Text file that will contain calculated values: {X Y Z} data1 data2 dataN som_x som_y cluster b_data1 b_data2 b_dataN in geospace.
geotiff_input=None      # geotiff_input
normalized="false"      # Whether the data has been normalized or not
minN=0                  # Minimum value for normalization
maxN=1                  # Maximum value for normalization
label=None              # Whether data contains label column, true or false


#------------------------------------------------------------------------------------------
#--- run SOM
#------------------------------------------------------------------------------------------

nxtsomcore = NxtSomCore()
if initialcodebook is not None: #if initial codebook was provided (in the form of som.dictionary), open the file, and load som codebook from it.
    with open(initialcodebook, 'rb') as som_dictionary_file:
        som_dictionary = pickle.load(som_dictionary_file)
        initialcodebook=som_dictionary['codebook']
        initialization=None           
header = nxtsomcore.load_data(input_file) 
som = nxtsomcore.train(
    header['data'],
    som_x,
    som_y,
    epochs,
    kerneltype=0,
    verbose=1,
    neighborhood=neighborhood,
    std_coeff=std_coeff,
    maptype=maptype,
    radiuscooling=radiuscooling,
    scalecooling=scalecooling,
    initialization=initialization,
    initialcodebook=initialcodebook,
    gridtype=gridtype
    )          
if(output_folder==""):
    output_folder="C:/Temp/NextSom"
else:
    output_folder=output_folder
print(output_folder)
if(kmeans=="true"):
    som['clusters']=nxtsomcore.clusters(som,kmeans_min,kmeans_max,kmeans_init,output_folder)     

if outgeofile is not None:
    nxtsomcore.save_geospace_result(outgeofile, header, som, output_folder, input_file, normalized, label) 

nxtsomcore.save_somspace_result(output_file_somspace, header, som, output_folder, input_file, normalized)  
if(geotiff_input is not None):
    inputFileArray=geotiff_input.split(",")    
    nxtsomcore.write_geotiff_out(output_folder, inputFileArray[0])
with open(output_folder+'/som.dictionary', 'wb') as som_dictionary_file:
    pickle.dump(som, som_dictionary_file) #save som object to file.
    


#------------------------------------------------------------------------------------------
#--- plot results
#------------------------------------------------------------------------------------------

