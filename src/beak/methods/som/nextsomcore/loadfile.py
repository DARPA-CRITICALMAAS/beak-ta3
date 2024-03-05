# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:10:46 2019

Script that servers as launch point to loading geotiff, csv or lrn files. Possibly more filetypes in the future

@author: shautala

Modyfied by Ina Storch, 2024
"""

from .lrnfile import load_lrn_file, read_lrn_coordinate_columns, read_lrn_data_columns
from ..load_geotiff import load_geotiff_files, delete_rows_with_no_data, read_geotiff_coordinate_columns, read_geotiff_data_columns
from ..load_csv import load_csv_file, read_csv_coordinate_columns,read_csv_data_columns


def load_input_file(input_file, label_file=""):#input file in case of lrn, input file list in case of geoTiff
    """load input file of type lrn, csv or geotiff

    Args:
        input_file (LiteralString): input file paths, separated by komma
        label_file (LiteralString, optional): label file paths. Defaults to "".

    Returns:
        dict: dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names
    """     
    if(input_file[-3:].lower()=="lrn"):#if input is lrn
        lrn_header=load_lrn_file(input_file)
        return lrn_header
    
    elif(input_file[-3:].lower()=="csv"):
        csv_header=load_csv_file(input_file)
        return csv_header
    else: 
        geotiff_header = load_geotiff_files(input_file, label_file)

        # Delete rows with noDataValue
        geotiff_header = delete_rows_with_no_data(geotiff_header)
        return geotiff_header
    

def read_coordinate_columns(header): 
    """read coordinate columns for file types lrn, csv and geotiff

    Args:
        header (dict): dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names

    Returns:
        dict: dictionary holding x and y coordinates as a column stack (ndarray), coordinates of rows with no data values that are skipped in load_input_file(), column names 
    """      
    if(header['filetype']=='lrn'):
        coords=read_lrn_coordinate_columns(header)    
        return coords
    elif(header['filetype']=='csv'):
        coords=read_csv_coordinate_columns(header)
        return coords
    else:
        coords=read_geotiff_coordinate_columns(header)
        return coords


def read_data_columns(header):
    """read data columns for file types lrn, csv and geotiff

    Args:
        header (dict): dictionary holding the data as a stacked ndarray and meta data such as geo transform, numer of rows and column, no data value, column (data) names

    Returns:
        dict: dictionary holding the data as a stacked ndarray and column (data) names
    """    
    if(header['filetype']=='lrn'):
        data=read_lrn_data_columns(header)
        return data
    elif(header['filetype']=='csv'):
        data=read_csv_data_columns(header)
        return data
    else:
        data=read_geotiff_data_columns(header)
        return data
    
