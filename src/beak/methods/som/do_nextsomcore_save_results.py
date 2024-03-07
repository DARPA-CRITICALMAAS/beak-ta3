from .nextsomcore.nextsomcore import NxtSomCore
import pickle
import time
import numpy as np

def run_SOM(args):
    """Load data, run SOM and k-means clustering, write output to file.

    Args:
        args (dict): Dictionary holding all parameters for SOM and k-means clustering.
    """    
    nxtsomcore = NxtSomCore()
    if args.initialcodebook is not None: #if initial codebook was provided (in the form of som.dictionary), open the file, and load som codebook from it.
        with open(args.initialcodebook, 'rb') as som_dictionary_file:
            som_dictionary = pickle.load(som_dictionary_file)
            args.initialcodebook=som_dictionary['codebook']
            args.initialization=None           
    
    print('Load data')
    # Record the start time
    start_time = time.time()
    if (args.label == True):
        header = nxtsomcore.load_data(args.input_file, args.label_geotiff_file) 
    else: 
        header = nxtsomcore.load_data(args.input_file) 
    # Record the end time
    end_time = time.time()
    # Print the elapsed time
    print(f"    Execution time: {end_time - start_time} seconds")

    print('Run SOM')
    start_time = time.time()
    som = nxtsomcore.train(
        header['data'],
        args.som_x,
        args.som_y,
        args.epochs,
        kerneltype=0,
        verbose=1,
        neighborhood=args.neighborhood,
        std_coeff=args.std_coeff,
        maptype=args.maptype,
        radiuscooling=args.radiuscooling,
        scalecooling=args.scalecooling,
        initialization=args.initialization,
        initialcodebook=args.initialcodebook,
        gridtype=args.gridtype
        )    
    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")

    if(args.output_folder==""):
        output_folder="C:/Temp/NextSom"
    else:
        output_folder=args.output_folder
    #print(args.output_folder)
    if(args.kmeans==True):
        start_time = time.time()
        som['clusters']=nxtsomcore.clusters(som,args.kmeans_min,args.kmeans_max,args.kmeans_init,output_folder)     
        end_time = time.time()
        print(f"    Execution time: {end_time - start_time} seconds")

    if args.outgeofile is not None:
        print('Save geo space results')
        start_time = time.time()
        nxtsomcore.save_geospace_result(args.outgeofile, header, som, output_folder, args.input_file, args.normalized, args.label) 
        end_time = time.time()
        print(f"    Execution time: {end_time - start_time} seconds")

    print('Save SOM space results')
    start_time = time.time()
    nxtsomcore.save_somspace_result(args.output_file_somspace, header, som, output_folder, args.normalized)  
    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")
    
    print('Save SOM object to file')
    start_time = time.time()
    with open(output_folder+'/som.dictionary', 'wb') as som_dictionary_file:
        pickle.dump(som, som_dictionary_file) #save som object to file.
    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")

    if(args.geotiff_input is not None):
        print('Write GeoTIFF file')
        start_time = time.time()
        inputFileArray=args.geotiff_input.split(",")    
        #nxtsomcore.write_geotiff_out(args.output_folder, inputFileArray[0])
        nxtsomcore.write_geotiff_out(args.output_folder, args.output_file_geospace, args.output_file_somspace, inputFileArray[0])
        end_time = time.time()
        print(f"    Execution time: {end_time - start_time} seconds")
    
    
    print('Count cluster hit count')
    start_time = time.time()
    
    cluster_hit_count(som, args.output_file_somspace, args.output_folder)
    
    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")


def cluster_hit_count(som, output_file_somspace, output_path):
    """Counts hits per cluster and creates a NumPy array with cluster numbers and hit counts. Saves the array to a text file.

    Args:
        som (dict): Dictionary holding the SOM codebook vectors, U-matrix, number of rows and columns, dimension and clusters.
        output_file_somspace (str): full file path for txt file holding som space results
        output_path (str): file path for writhing txt file
    """
    som_data = np.genfromtxt(output_file_somspace, skip_header=(1), delimiter=' ')
    clusters=int(max(som_data[:,len(som_data[0])-2])+1)

    # Initialize a list to store hit counts for each cluster
    cluster_hit_count = [0] * (clusters)  # Initialize with zeros

    if(clusters>1):
        cluster_array=som['clusters'].transpose()
        for i in range (clusters,0,-1):
            for bmu in som['bmus']:
                if (cluster_array[bmu[0]][bmu[1]])+1==i:
                    cluster_hit_count[i-1]+=1
            print(f"        Cluster hit count: {i-1} - {cluster_hit_count[i-1]}")

    # Create a NumPy array with cluster numbers and hit counts
    result_array = np.column_stack((np.arange(clusters), cluster_hit_count))

    # Save the array to a text file
    np.savetxt(output_path+"/cluster_hit_count.txt", result_array, fmt='%d', delimiter='\t', header='ClusterNumber\tHitCount', comments='')
