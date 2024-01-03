from .nextsomcore.nextsomcore import NxtSomCore
import pickle

def run_SOM(args):
    nxtsomcore = NxtSomCore()
    if args.initialcodebook is not None: #if initial codebook was provided (in the form of som.dictionary), open the file, and load som codebook from it.
        with open(initialcodebook, 'rb') as som_dictionary_file:
            som_dictionary = pickle.load(som_dictionary_file)
            args.initialcodebook=som_dictionary['codebook']
            args.initialization=None           
    print('Load data')
    header = nxtsomcore.load_data(args.input_file) 
    print('Run SOM')
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
    if(args.output_folder==""):
        output_folder="C:/Temp/NextSom"
    else:
        output_folder=args.output_folder
    print(args.output_folder)
    if(args.kmeans.lower()=="true"):
        som['clusters']=nxtsomcore.clusters(som,args.kmeans_min,args.kmeans_max,args.kmeans_init,output_folder)     
    
    if args.outgeofile is not None:
        print('Save geo space results')
        nxtsomcore.save_geospace_result(args.outgeofile, header, som, output_folder, args.input_file, args.normalized, args.label) 
    
    print('Save SOM space results')
    nxtsomcore.save_somspace_result(args.output_file_somspace, header, som, output_folder, args.normalized)  
    if(args.geotiff_input is not None):
        print('Write GeoTIFF file')
        inputFileArray=args.geotiff_input.split(",")    
        #nxtsomcore.write_geotiff_out(args.output_folder, inputFileArray[0])
        nxtsomcore.write_geotiff_out(args.output_folder, args.output_file_geospace, args.output_file_somspace, inputFileArray[0])
    with open(output_folder+'/som.dictionary', 'wb') as som_dictionary_file:
        pickle.dump(som, som_dictionary_file) #save som object to file.
    
    print("noDataValue: ", header['noDataValue'])  