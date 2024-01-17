from .nextsomcore.nextsomcore import NxtSomCore
import pickle
import time
import numpy as np

def run_SOM(args):
    nxtsomcore = NxtSomCore()
    if args.initialcodebook is not None: #if initial codebook was provided (in the form of som.dictionary), open the file, and load som codebook from it.
        with open(initialcodebook, 'rb') as som_dictionary_file:
            som_dictionary = pickle.load(som_dictionary_file)
            args.initialcodebook=som_dictionary['codebook']
            args.initialization=None           
    
    print('Load data')
    # Record the start time
    start_time = time.time()
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
    if(args.kmeans.lower()=="true"):
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
    
    #print("noDataValue: ", header['noDataValue'])  


    print('Count hits per SOM cell')
    start_time = time.time()
    som_data = np.genfromtxt(args.output_file_somspace,skip_header=(1), delimiter=' ')

    clusters=int(max(som_data[:,len(som_data[0])-2])+1)

    # Initialize a list to store hit counts for each cluster
    cluster_hit_count = [0] * (clusters)  # Initialize with zeros

    #labeling clusters in colorbar with format "cluster number:  number of data points in this cluster".
    if(clusters>1):
        cluster_array=som['clusters'].transpose()#TODO: figure out if this a problem elsewhere.
        for i in range (clusters,0,-1):
            for bmu in som['bmus']:
                if (cluster_array[bmu[0]][bmu[1]])+1==i:
                    cluster_hit_count[i-1]+=1
            print(f"        Cluster hit count: {i-1} - {cluster_hit_count[i-1]}")
    
    end_time = time.time()
    print(f"    Execution time: {end_time - start_time} seconds")