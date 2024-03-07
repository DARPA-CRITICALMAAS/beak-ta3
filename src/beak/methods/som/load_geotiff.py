# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:46:21 2019
Script for reading geotiff input data files.
@author: shautala
"""

import numpy as np
from osgeo import gdal
gdal.DontUseExceptions()  # gdal.UseExceptions()  # Explicitly Use or Don't Use Exceptions
import os
import sys

def load_geotiff_files(input_file_list, label_file=""):
    """Load data and meta data of geotiff files using gdal

    Args:
        input_file_list (LiteralString): input file paths, separated by komma
        label_file (LiteralString, optional): label file paths, separated by komma. Defaults to "".

    Returns:
        dict: dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names
    """    
    width_0 = None
    height_0 = None
    gt_0 = None
    prj_0=None
    path_0=None
    noDataValue_0 = None
    dataType_0 = None
    returndata = np.array([]) 
    labeldata = np.array([]) 
    label = False
    colnames=[]
    colnames_label = []
    geotiff_list_as_string=input_file_list # At this points the assumption is made that the coordinates of separate files are identical. So the coordinates can just be taken from any one of the individual files.

    geotiff_array=geotiff_list_as_string.split(",")    #geotiff_list is str. geotiff_list_2 is array #maybe change the parameter name...?
    layer_count = len(geotiff_array)

    # if file with label data is provided, add to list
    if(label_file!=""):
        label = True
        geotiff_array += label_file.split(",")

    for geotiffpath in geotiff_array:   
        src_ds = gdal.Open(geotiffpath, gdal.GA_ReadOnly)
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
        #if noDataValue != noDataValue_0:
        #    print("Warning: noDataValue of " + geotiffpath + " does not match with " + path_0)
        #if dataType != dataType_0:
        #    print("Warning: dataType of " + geotiffpath + " does not match with " + path_0)

    	# Read the raster data as a NumPy array
        raster_array = src_ds.ReadAsArray()

        # Set NoData values to np.nan
        raster_array = np.where(raster_array == noDataValue, np.nan, raster_array)

        # Check if data type is integer and convert to np.float32
        if raster_array.dtype.kind == 'i':
            raster_array = raster_array.astype(np.float32)

        # Close the dataset
        src_ds = None

        flat=raster_array.flatten(order='C')
        

        if(len(colnames) < layer_count):
            # Use np.concatenate to stack arrays
            returndata = np.concatenate([returndata, flat]) if returndata.size else flat
            colnames.append(os.path.basename(geotiffpath))     
        else:
            # Use np.concatenate to stack arrays
            labeldata = np.concatenate([labeldata, flat]) if labeldata.size else flat
            #colnames.append("label")
            colnames_label.append("label_" + os.path.basename(geotiffpath))

    # Reshape returndata to have each raster as a column
    returndata = returndata.reshape(-1, layer_count, order='F')
    labeldata = labeldata.reshape(-1, len(geotiff_array) - layer_count, order='F')
    rows=len(returndata) 
    cols=len(returndata[0])              

    return {'rows': rows, 'cols': cols, 'colnames': colnames, 'label': label, 'labeldata': labeldata, 'colnames_label': colnames_label,
            'headerlength': 0, 'data': returndata, 'filetype': 'geotiff','originaldata':raster_array, 
            'geotransform':gt, 'noDataValue':np.nan, 'dataType':dataType
            }   

def delete_rows_with_no_data(geotiff_header):
    """Delete rows from ndarray that contain noData values. Data and rows count of ndarray is updated and returned as a dictionary with same format as input dictionary.

    Args:
        geotiff_header (dict): dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names

    Returns:
        dict: dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names
    """    
    data = geotiff_header['data']
    labeldata = geotiff_header['labeldata']

    # Identify rows with noDataValue
    rows_to_delete = np.any(np.isnan(data), axis=1)

    # Delete rows with noDataValue
    data_filtered = data[~rows_to_delete]
    if(geotiff_header['label']==True): 
        labeldata_filtered = labeldata[~rows_to_delete]
    else:
        labeldata_filtered = None

    # Update the rows count
    rows = len(data_filtered)

    return {'rows': rows, 'cols': geotiff_header['cols'], 'colnames': geotiff_header['colnames'], 'label': geotiff_header['label'], 'labeldata_all': geotiff_header['labeldata'],'labeldata': labeldata_filtered, 'colnames_label': geotiff_header['colnames_label'],
            'headerlength': 0, 'data_all': geotiff_header['data'], 'data': data_filtered, 'filetype': 'geotiff', 'originaldata': geotiff_header['originaldata'],
            'geotransform': geotiff_header['geotransform'], 'noDataValue': geotiff_header['noDataValue'], 'dataType': geotiff_header['dataType']
            }

def read_geotiff_coordinate_columns(geotiff_header):
    """Read coordinate columns from dictionary

    Args:
        geotiff_header (dict): dictionary holding the data as a ndarray and meta data, such as numer of rows and column, column (data) names, file type

    Returns:
        dict: dictionary holding coordinates as a ndarray, column names and format
    """    
    coordinates_x=[]
    coordinates_y=[]
    coordinates_deleted = []         
    data=geotiff_header['originaldata']
    gt=geotiff_header['geotransform']   # gt[0]: x upper left, gt[1]: dx, gt[2]: row rotation (typically =0)
                                        # gt[3]: y upper left, gt[4]: column rotation (typically =0), gt[5]: dy (negative value for a north-up image) 
    
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            coordinates_x.append(j*gt[1]+gt[0])
            coordinates_y.append(i*gt[5]+gt[3])
    coordinates_all=np.column_stack((coordinates_x,coordinates_y))

    colnames = ['X', 'Y']

    if('data_all' in geotiff_header):        
        # Identify rows with noDataValue
        rows_to_delete = np.any(np.isnan(geotiff_header['data_all']), axis=1)

        # Delete rows with noDataValue
        coordinates = coordinates_all[~rows_to_delete]
        coordinates_deleted = coordinates_all[rows_to_delete]
    else:  
        coordinates = coordinates_all

    return {'data': coordinates, 'data_deleted': coordinates_deleted, 'data_all':coordinates_all,'colnames': colnames, 'fmt': '%f %f'} 

def read_geotiff_data_columns(geotiff_header):
    """Read data columns from dictionary

    Args:
        geotiff_header (dict): dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names

    Returns:
        dict: dictionary holding data columns as a ndarray, column names and format
    """    
    return {'data': geotiff_header['data'], 'colnames': geotiff_header['colnames'], 'fmt': ('%f ' * geotiff_header['cols']).rstrip()} 



