# -*- coding: utf-8 -*-
"""
The module contains the NxtSomCore class that is used to train self-organizing
maps and write results to the disk. NxtSomCore depends on somoclu package written
by Peter Wittek to perform actual SOM calculations.

@author: Janne Kallunki, Sakari Hautala
modyfied by Ina Storch, 2024
"""
import warnings
with warnings.catch_warnings():
    import numpy as np
    from sklearn.cluster import KMeans
    import somoclu
    import sys
    #from .lrnfile import load_lrn_file, read_coordidate_columns, read_data_columns
    from .loadfile import load_input_file, read_coordinate_columns, read_data_columns
    import heapq
    import pickle
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from sklearn.metrics import davies_bouldin_score
    from decimal import Decimal
    import os.path
    #import ast
class NxtSomCore(object):
    """Class for training self-organizing map and saving results.
    """
    def __init__(self):
        """Constructor for the class
        """
        self.som = {}

    def load_data(self, input_file, label_file = ""):
        """Load and return the input data as a dict containing numpy array and metadata

        Args:
            input_file (str): The name of the file to be loaded.
            label_file (str, optional): The name of the label file to be loaded. Defaults to "".

        Returns:
            dict: Dictionary holding de model data as a ndarray and meta data such as numer of rows and column, column (data) names, file type. 
        """        

        return load_input_file(input_file, label_file)

    def train(self, data, som_x, som_y , epochs=10, **kwargs):
        """Train the map and return results as a dict.

        Args:
            data (nparray): Training data used in SOM.
            som_x (int): X-size of the map.
            som_y (int): Y-size of the map.
            epochs (int, optional):  Number of rounds the training is performed. Defaults to 10.

        Returns:
            dict: Dictionary holding SOM results, umatrix and meta data like number of columns and rows.
        """        
        
        self.som = somoclu.Somoclu(som_x, som_y, 
            kerneltype = kwargs.pop("kerneltype", 0), 
  			verbose = kwargs.pop("verbose", 2),
  			neighborhood = kwargs.pop("neighborhood", "gaussian"), 
  			std_coeff = kwargs.pop("std_coeff", 0.5),  
            maptype= kwargs.pop("maptype", "toroid"), 
            initialcodebook=kwargs.pop("initialcodebook", None), 
            initialization=kwargs.pop("initialization", 'random'), 
            gridtype=kwargs.pop("gridtype","rectangular" ) )
        self.som.train(data, epochs,
			radius0 = kwargs.pop("radius0", 0),
			radiusN = kwargs.pop("radiusN", 1),
			radiuscooling = kwargs.pop("radiuscooling", "linear"),
			scale0 = kwargs.pop("scale0", 0.1),
			scaleN = kwargs.pop("scaleN", 0.01),
			scalecooling = kwargs.pop("scalecooling", "linear"))
        return {
            'codebook': self.som.codebook.copy(),
            'bmus': self.som.bmus.copy(),
            'umatrix' : self.som.umatrix.copy(),
            'n_columns' : self.som._n_columns,
            'n_rows': self.som._n_rows,
            'n_dim': self.som.n_dim,
            'clusters': None}

    def cluster(self, som, cluster_count):
        """Cluster the codebook and return clustering results as a 2d numpy array. Code taken from
         somoclu's train.py and changed to operate on input parameters only

        Args:
            som (dict): SOM-related data obtained from training(codebook, dimensions, etc..)
            cluster_count (int): Number of clusters used in clustering

        Returns:
            nparray: Cluster result.
        """        

        algorithm = KMeans(n_clusters=cluster_count, init='random', n_init=10) # n_init='auto'
        original_shape = som['codebook'].shape
        som['codebook'].shape = (som['n_columns'] * som['n_rows'], som['n_dim'])
        linear_clusters = algorithm.fit_predict(som['codebook'])
        som['codebook'].shape = original_shape
        clusters = np.zeros((som['n_rows'], som['n_columns']), dtype=int)
        for i, c in enumerate(linear_clusters):
            clusters[i // som['n_columns'], i % som['n_columns']] = c
        som['clusters'] = clusters
        return clusters


    def clusters(self, som, cluster_min, cluster_max, cluster_init,working_dir): 
        """Cluster the codebook and return clustering results as a 2d numpy array. Code taken from
         somoclu's train.py and changed to operate on input parameters only.
         Calculates clusters multiple times and selects the best result by lowest Davies-Bouldin score. 
         Returns the best clustering, and writes the best clustering results for each number of clusters to a binary file.

        Args:
            som (dict): SOM-related data obtained from training(codebook, dimensions, etc..)
            cluster_min (int): Minimum number of clusters used in clustering
            cluster_max (int): Maximum number of clusters used in clustering
            cluster_init (int): Number of initializations used in clustering
            working_dir (str): Destination folder to be used for saving the result.

        Returns:
            nparray: Cluster result with smalles db-score.
        """        
             
        min=2 
        algorithm = KMeans()
        original_shape = som['codebook'].shape       
        if (cluster_init<1):
            cluster_init=1
        if(cluster_min<2):
            cluster_min=2   
        if(cluster_max<3):
            cluster_max=3
        cluster_list=[]    
        total=(cluster_max-cluster_min+1)*cluster_init 
        value=0
        if(total>20):
            interval=int(total/10)
        else:
            interval=1      
        
        print("Clustering progress:")
        for a in range(cluster_min, cluster_max+1):                                   
            min=float("inf")
            min_dict={}
            for j in range(0, cluster_init):
                algorithm = KMeans(n_clusters=a,init='random', n_init=10)
                som['codebook'].shape = (som['n_columns'] * som['n_rows'], som['n_dim'])  
                linear_clusters=algorithm.fit_predict(som['codebook'])       
                current=davies_bouldin_score(som['codebook'], linear_clusters)
                clusters = np.zeros((som['n_rows'], som['n_columns']), dtype=int)
                for i, c in enumerate(linear_clusters):
                    clusters[i // som['n_columns'], i % som['n_columns']] = c
                som['codebook'].shape = original_shape
                dict={'db_score': current, 'n_clusters' : a, 'cluster' : clusters}
                
                value=((a-cluster_min)*cluster_init + j)
                if (value%interval==0):
                    print(("%.2f" % ((value/total)*100))+"%")
                if(min>current):
                    min=current
                    #min_clusters=clusters
                    min_dict=dict
            cluster_list.append(min_dict)
        smallest_3=[]
        smallest_3=heapq.nsmallest(3, cluster_list,key=lambda k: k['db_score'])    
        with open(working_dir+'/cluster.dictionary', 'wb') as cluster_dictionary_file:
            pickle.dump(cluster_list, cluster_dictionary_file)
        print("100% Clustering completed.")
        return smallest_3[0]["cluster"]     


    def save_geospace_result(self, output_file, header, som, output_folder, input_file, normalized=False, labelIndex=False):
        """Write SOM results with header line and input columns to disk in geospace.

        Output file columns:
        X, Y, Z - Coordinates in the input file (if present in input file)
        data1, data2, dataN, ... - Original columns in input file
        som_x, som_y - X Y indices in the map
        cluster - Cluster group number
        b_data1, bdata2, bdataN, ... - SOM results
        q_error= Q-error, difference between som results and original columns

        Args:
            output_file (str): Filename to be used for saving the result.
            header (dict): Dictionary holding header data ((load_data(...)).
            som (dict): Dictionary holding SOM results (train(...)).
            output_folder (str): Destination folder to be used for saving the result.
            input_file (LiteralStringstr): list of input files, separated by komma (only first line is used to get the geotransform and projection information to set output GeoFIT geotransform and projection)
            normalized (bool, optional): Whether data is normalized or not. Defaults to False.
            labelIndex (bool, optional): Whether input data has label data column. Defaults to False.
        """        
        
        coord_cols = read_coordinate_columns(header)
        data_cols = read_data_columns(header)
        som_cols = self._extract_som_cols_geospace(som, data_cols['colnames'])   

        q_cols = som_cols['data'][:,3:] - data_cols['data'] #calculate q error
        q_error=np.linalg.norm(q_cols,axis=1)               #calculate q error     
        mean_q_error=np.mean(q_error)                       #calculate mean q error

        with open(output_folder + "/RunStats.txt", "a+") as f:      #write mean q error to runstats file
            f.write("Quantization error: " + str(mean_q_error))

        xml_file = Path(output_folder+"/RunStats.xml")

        if xml_file.is_file():
            tree = ET.parse(xml_file)
            root = tree.getroot()
            q_error_mean = ET.Element("q_error")
            q_error_mean.text = str(mean_q_error)
            root.append(q_error_mean)
            tree.write(xml_file)
        
        header_line = '{} {} {} {}'.format(str(coord_cols['colnames']),
										str(som_cols['colnames']),
										str(data_cols['colnames']),'q_error'
                                        ).replace('[','').replace(']','').replace(',','').replace('\'','')
        
        if normalized == "true":
            data_cols["data"] = data_cols["data"].astype('float64')
            tree = ET.parse(output_folder + "/DataStats.xml")
            root = tree.getroot()
            for i in range(0, len(data_cols["data"][0])):
                maxD = Decimal(tree.find(data_cols["colnames"][i]).find("max").text)
                minD = Decimal(tree.find(data_cols["colnames"][i]).find("min").text)
                minN = float(Decimal(tree.find(data_cols["colnames"][i]).find("scaleMin").text))
                maxN = float(Decimal(tree.find(data_cols["colnames"][i]).find("scaleMax").text))
                for j in range(0, len(data_cols["data"])):
                    N = Decimal(data_cols["data"][j][i].item())
                    data_cols["data"][j][i] = (maxD - minD) * (N - Decimal(minN)) / (Decimal(maxN) - Decimal(minN)) + minD
                    N = Decimal(som_cols["data"][j][i + 3].item())
                    som_cols["data"][j][i + 3] = (maxD - minD) * (N - Decimal(minN)) / (Decimal(maxN) - Decimal(minN)) + minD
        
        print("         combine data colums for output geo file (for large data arrays memory usage might be a concern)")

        combined_cols = np.c_[coord_cols['data'], som_cols['data'], data_cols['data'], q_error]     
        combined_cols_deleted = np.c_[coord_cols['data_deleted'], np.full((coord_cols['data_deleted'].shape[0],combined_cols.shape[1] - 2), np.nan)]

        # Join the arrays
        final_combined_cols = np.vstack((combined_cols, combined_cols_deleted))
        
        print("         savetxt")

        if(labelIndex==True):
            data = np.loadtxt(
                input_file, 
                dtype='str',
                delimiter='\t',
                skiprows=3
            )
            labelcol=[]
            for i in range(0,len(data[0])):
                if(data[0][i]=='label'):
                    labelcol=data[1:,i]
            combined_cols=np.c_[combined_cols,labelcol]
            combined_cols_deleted=np.c_[combined_cols_deleted,np.full((coord_cols['data_deleted'].shape[0],1), np.nan)]
            final_combined_cols = np.vstack((combined_cols, combined_cols_deleted))
            header_line= header_line+" label"            
            np.savetxt(output_file, final_combined_cols,fmt='%s', header=header_line, delimiter=' ', comments='')
            np.savetxt(output_file[:-3]+"csv", final_combined_cols,fmt='%s', header=header_line.replace(" ",","),delimiter=',', comments='')
            #np.savetxt(output_file[:-3] + "csv", final_combined_cols, fmt=header_line.replace(" ", ","), delimiter=',', comments='')
        else:            
            fmt_combined = '{} {} {} {}'.format(coord_cols['fmt'], som_cols['fmt'], data_cols['fmt'], '%.5f')#'%.5f')       
            np.savetxt(output_file, final_combined_cols, fmt='%s', header=header_line, delimiter=' ', comments='')
            #np.savetxt(output_file, final_combined_cols,fmt=fmt_combined, header=header_line, delimiter=' ', comments='')
            np.savetxt(output_file[:-3] + "csv", final_combined_cols, fmt=fmt_combined.replace(" ",","), header=header_line.replace(" ",","), comments='')
            #np.savetxt(output_file[:-3] + "csv", final_combined_cols, fmt=header_line.replace(" ", ","), delimiter=',', comments='')

    def save_somspace_result(self, output_file, header, som, output_folder, normalized=False):
        """Write SOM results with header line and input columns to disk in somspace.

        Output file columns:
        som_x, som_y - X Y indices in the map
        b_data1, bdata2, bdataN, ... - SOM results
        umatrix - U-matrix
        cluster - Cluster group number

        Args:
            output_file (str): Filename to be used for saving the result.
            header (dict): Dictionary holding header data ((load_data(...)).
            som (dict): Dictionary holding SOM results (train(...)).
            output_folder (str): Destination folder to be used for saving the result.
            normalized (bool, optional): Whether data is normalized or not. Defaults to False.
        """        

        col_names = read_data_columns(header)['colnames']
        som_cols = self._extract_som_cols_somspace(som, col_names)
        hits=np.zeros((som['n_columns'],som['n_rows']))

        for i in range(0,len(som["bmus"])):
             hits[som["bmus"][i][0]][som["bmus"][i][1]]+=1
        hits=hits.flatten()

        if(normalized=="True"):
            data_cols = read_data_columns(header)
            tree = ET.parse(output_folder+"/DataStats.xml")
            for i in range(2,len(som_cols["data"][0])-2):               	
                    maxD=Decimal(tree.find(data_cols["colnames"][i-2].replace("\"","")).find("max").text)
                    minD=Decimal(tree.find(data_cols["colnames"][i-2].replace("\"","")).find("min").text)        
                    minN=float(Decimal(tree.find(data_cols["colnames"][i-2].replace("\"","")).find("scaleMin").text))
                    maxN=float(Decimal(tree.find(data_cols["colnames"][i-2].replace("\"","")).find("scaleMax").text)) 
                    for j in range(0,len(som_cols["data"])):
                        N=Decimal(som_cols["data"][j][i].item())
                        som_cols["data"][j][i]=(maxD-minD)*(N-Decimal(minN))/(Decimal(maxN)-Decimal(minN))+minD   

        som_cols["data"]=np.hstack((som_cols["data"],hits.reshape(-1,1)))
        som_cols['fmt']=som_cols['fmt']+ " %f"
        header_line = '{}'.format(str(som_cols['colnames'])).replace('[','').replace(']','').replace(',','').replace('\'','')+" hits"#.translate(None, "[]',") replaced by a lower level solution that works in both 2.x and 3.x python
        np.savetxt(output_file, som_cols['data'], fmt=som_cols['fmt'], header=header_line, comments='')
        np.savetxt(output_file[:-3]+"csv", som_cols['data'], fmt=som_cols['fmt'].replace(" ",","), header=header_line.replace(" ",","), comments='')

    def _extract_som_cols_geospace(self, som, col_names):
        x_col = som['bmus'][:, 1]
        y_col = som['bmus'][:, 0]    
        data = som['codebook'][x_col, y_col]
        if (som['clusters'] is not None):            
            clusters = som['clusters'][x_col, y_col]
            combined_data = np.c_[y_col, x_col, clusters, data]
            combined_col_list = ['som_x', 'som_y', 'cluster'] + ['b_%s' % x for x in col_names]
            combined_fmt = ('%f %f %f ' + '%5f ' * len(col_names)).rstrip()
        else:
            clusters=np.zeros(shape=(len(x_col)))
            combined_data = np.c_[y_col, x_col, clusters, data]
            combined_col_list = ['som_x', 'som_y', 'cluster'] + ['b_%s' % x for x in col_names]
            combined_fmt = ('%d %d %d ' + '%.5f ' * len(col_names)).rstrip()
        return {'data': combined_data, 'colnames':combined_col_list, 'fmt': combined_fmt}


    def _extract_som_cols_somspace(self, som, col_names):  
        rows = som['codebook'].shape[0]
        cols = som['codebook'].shape[1]
        som_x_y_cols = np.array(np.meshgrid(np.arange(cols), np.arange(rows))).T.reshape(-1, 2)
        row = som_x_y_cols[:, 0]
        col = som_x_y_cols[:, 1]
        data = som['codebook'][col, row]#col=x, row=y
        umatrix = som['umatrix'][col, row]
        if (som['clusters'] is not None):
            clusters = som['clusters'][col, row]
            combined_data = np.c_[som_x_y_cols, data, umatrix, clusters]
            combined_col_list = ['som_x', 'som_y'] + ['b_%s' % x for x in col_names] + ['umatrix', 'cluster']
            combined_fmt = ('%d %d ' + '%f ' * len(col_names)) +'%f %f'
        else:           
            clusters=np.zeros(shape=(len(umatrix)))
            combined_data = np.c_[som_x_y_cols, data, umatrix, clusters]
            combined_col_list = ['som_x', 'som_y'] + ['b_%s' % x for x in col_names] + ['umatrix', 'cluster']
            combined_fmt = ('%d %d ' + '%f ' * len(col_names)) +'%f %f'
        return {'data': combined_data, 'colnames':combined_col_list, 'fmt': combined_fmt}


    def write_geotiff_out(self, output_folder, geodatafile, somdatafile, input_file): 
        """Write geotiff to file using gdal.

        Args:
            output_folder (str): folder path where to write geotiff file
            geodatafile (str): path and name for text file that contains som results in geospace
            somdatafile (str): path and name for text file that contains som results in somspace
            input_file (LiteralStringstr): list of input files, separated by komma (only first line is used to get the geotransform and projection information to set output GeoFIT geotransform and projection)
        """        
        from osgeo import gdal
        import pandas as pd

        #print("     input file: ", input_file)

        inDs=gdal.Open(input_file.split(',')[0])
        inBand=inDs.GetRasterBand(1)
        gt = inDs.GetGeoTransform()

        #print("     genfromtxt som_data")
        #som_data= np.genfromtxt(somdatafile,skip_header=(1), delimiter=' ')
        #print("     genfromtxt geo_data")
        #geo_data=np.genfromtxt(geodatafile,skip_header=(1), delimiter=' ') 
        #headers=[]

        print("     read_csv som_data")
        som_data = pd.read_csv(somdatafile, skiprows=1, delimiter=' ').values
        print("     read_csv geo_data")
        geo_data = pd.read_csv(geodatafile, skiprows=1, delimiter=' ').values
        headers = pd.read_csv(geodatafile, nrows=0, delimiter=' ').columns.tolist()

        x=geo_data[:,0]
        y=geo_data[:,1]

        # Create the destination folder if it doesn't exist
        destination_path = output_folder + "/GeoTIFF"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        #print("     open(geodatafile)")
        #
        #with open(geodatafile) as gd:   # easier way using pandas?
        #    line = gd.readline()
        #    headers=line.split()

        print("     Iterate over each geoTIF file:")

        for a in range(0, som_data.shape[1]-4): 
            print("         ", os.path.splitext(os.path.basename(headers[4 + a]))[0])

            z=geo_data[:,(4+a)]
            df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
            df.columns = ['X_value','Y_value','Z_value']
            df['Z_value'] = pd.to_numeric(df['Z_value'])
            pivotted= df.pivot(index='Y_value',columns='X_value',values='Z_value')

            cols = pivotted.shape[1]
            rows = pivotted.shape[0]

            driver = gdal.GetDriverByName('GTiff')

            # Create the output GeoTIFF
            outName = destination_path + "/out_" + os.path.splitext(os.path.basename(headers[4 + a]))[0] + ".tif"
            outDs = driver.Create(outName, cols, rows, 1, gdal.GDT_Float32)

            if outDs is None:
                print ("Could not create tif file")
                sys.exit(1) 
    
            outBand = outDs.GetRasterBand(1)
            #outData = pivotted.to_numpy()
            outData = np.flip(pivotted.to_numpy(), 0) 

            outBand.WriteArray(outData, 0, 0)
            outBand.FlushCache()
            outBand.SetNoDataValue(inBand.GetNoDataValue())
            #outBand.SetNoDataValue(-99)

            # Use the geotransform and projection information from the input file
            outDs.SetGeoTransform(inDs.GetGeoTransform())
            outDs.SetProjection(inDs.GetProjection())

            # check
            #print("     Name geoTIF output: ", outName)

            # Flush and close the output dataset
            outDs.FlushCache()
            outDs = None
            
        #q_error. output columns should probably be rearranged, so these could be handled in a single for loop again. or split main code into a different function
        #x=geo_data[:,0]
        #y=geo_data[:,1]
        z=geo_data[:,(len(som_data[0])-5)*2 +5]
        df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
        df.columns = ['X_value','Y_value','Z_value']
        df['Z_value'] = pd.to_numeric(df['Z_value'])
        pivotted= df.pivot(index='Y_value',columns='X_value',values='Z_value')

        driver = gdal.GetDriverByName('GTiff')
        outName = destination_path + "/out_q_error.tif"
        outDs = driver.Create(outName, cols, rows, 1, gdal.GDT_Float32)

        if outDs is None:
            print ("Could not create tif file")
            sys.exit(1) 

        outBand = outDs.GetRasterBand(1)
        #outData = pivotted.to_numpy()
        outData = np.flip(pivotted.to_numpy(), 0)  
        
        outBand.WriteArray(outData, 0, 0)
        outBand.FlushCache()
        outBand.SetNoDataValue(inBand.GetNoDataValue())

        outDs.SetGeoTransform(inDs.GetGeoTransform())
        outDs.SetProjection(inDs.GetProjection())
        
        outDs.FlushCache()
        inDs=None
        outDs=None
            
    
