# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:46:21 2019
Script for reading geotiff input data files.
@author: shautala
"""

import numpy as np
from osgeo import gdal
import os
import sys

def load_geotiff_files(input_file_list):
    
    width_0 = None
    height_0 = None
    gt_0 = None
    prj_0=None
    path_0=None
    noDataValue_0 = None
    dataType_0 = None
    geotiff_list_as_string=input_file_list # At this points the assumption is made that the coordinates of separate files are identical. So the coordinates can just be taken from any one of the individual files.
    returndata=[]
    colnames=[]

    if("," in geotiff_list_as_string):
        geotiff_list_2=geotiff_list_as_string.split(",")#geotiff_list is str. geotiff_list_2 is array #maybe change the parameter name...?
    else:
        geotiff_list_2=[geotiff_list_as_string]

    for geotiffpath in geotiff_list_2: 
        src_ds = gdal.Open(geotiffpath)   
        band=src_ds.GetRasterBand(1)
        noDataValue = band.GetNoDataValue()
        dataType = band.DataType  
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        gt = src_ds.GetGeoTransform()
        prj=src_ds.GetProjection()

        if(width_0 is None):        #stash baseline values on first loop, for checking that projection etc. match for all files
            width_0 = width
            height_0 = height
            gt_0 = gt
            prj_0=prj
            path_0=geotiffpath
            noDataValue_0 = noDataValue
            dataType_0 = dataType

        if(gt!=gt_0):
            print("Warning: Geotransform of "+geotiffpath+" does not match with "+path_0)
        if(prj!=prj_0):
            print("Warning: Projection of "+geotiffpath+" does not match with "+path_0)
        if(width!=width_0 or height!=height_0):
            sys.exit("Error: Grid of "+geotiffpath+" does not match."+path_0)
        if noDataValue != noDataValue_0:
            print("Warning: noDataValue of " + geotiffpath + " does not match with " + path_0)
        if dataType != dataType_0:
            print("Warning: dataType of " + geotiffpath + " does not match with " + path_0)

        data = src_ds.ReadAsArray()
        flat=data.flatten(order='C')
        
        if(len(returndata)>0):
            returndata=np.column_stack((returndata,flat))
        else:
            returndata=flat
        colnames.append(os.path.basename(geotiffpath))            
    rows=len(returndata)   
    if("," in geotiff_list_as_string):
        cols=len(returndata[0])    
    else:
        cols=1            
    #band=src_ds.GetRasterBand(1)
    #noDataValue= band.GetNoDataValue()
    #dataType=band.DataType
    return {'rows': rows, 'cols': cols, 'colnames': colnames, 
            'headerlength': 0, 'data': returndata, 'filetype': 'geotiff','originaldata':data, 
            'geotransform':gt, 'noDataValue':noDataValue, 'dataType':dataType
            }   

def delete_rows_with_no_data(geotiff_header):
    data = geotiff_header['data']
    originaldata = geotiff_header['originaldata']
    noDataValue = geotiff_header['noDataValue']

    # Identify rows with noDataValue
    rows_to_delete = np.any(data == noDataValue, axis=1)

    # Delete rows with noDataValue
    data_filtered = data[~rows_to_delete]

    # Update the rows count
    rows = len(data_filtered)

    return {'rows': rows, 'cols': geotiff_header['cols'], 'colnames': geotiff_header['colnames'],
            'headerlength': 0, 'data_all': geotiff_header['data'], 'data': data_filtered, 'filetype': 'geotiff', 'originaldata': originaldata,
            'geotransform': geotiff_header['geotransform'], 'noDataValue': noDataValue, 'dataType': geotiff_header['dataType']
            }

def read_geotiff_coordinate_columns(geotiff_header):
    coordinates_x=[]
    coordinates_y=[]         
    data=geotiff_header['originaldata']
    gt=geotiff_header['geotransform'] #gt[0], gt[1], gt[3] and gt[4] have all that is needed 
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            coordinates_x.append(j*gt[1]+gt[0])
            coordinates_y.append(i*gt[5]+gt[3])
    coordinates=np.column_stack((coordinates_x,coordinates_y)) #Coordinates are just indexes at this stage. TODO: use gt to tranform them back into real world values.
    colnames = ['X', 'Y']

    if('data_all' in geotiff_header):
        # Identify rows with noDataValue
        noDataValue = geotiff_header['noDataValue']
        rows_to_delete = np.any(geotiff_header['data_all'] == noDataValue, axis=1)

        # Delete rows with noDataValue
        coordinates = coordinates[~rows_to_delete]

    return {'data': coordinates, 'colnames': colnames, 'fmt': '%f %f'} 

def read_geotiff_data_columns(geotiff_header):
    return {'data': geotiff_header['data'], 'colnames': geotiff_header['colnames'], 'fmt': ('%f ' * geotiff_header['cols']).rstrip()} 



