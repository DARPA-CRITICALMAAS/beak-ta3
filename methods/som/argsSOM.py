class Args:
    input_file: str            # Input file(*.lrn)
    input_list_text=[]         # only for geotiff files: input_file= ",".join(input_list_text)
    output_folder: str         # Folder to save som dictionary and cluster dictionary
    output_file_somspace: str   
    outgeofile: str   

    som_x: int                # X dimension of generated SOM
    som_y: int                # Y dimension of generated SOM
    epochs: int               # Number of epochs to run

    maptype='toroid'            # Type of SOM (sheet, toroid)
    initialcodebook=None        # File path of initial codebook, 2D numpy.array of float32.
    neighborhood='gaussian'     # Shape of the neighborhood function. gaussian or bubble
    std_coeff=0.5               # Coefficient in the Gaussian neighborhood function
    radius0=0.0                   # Initial size of the neighborhood
    radiusN=1.0                   # Final size of the neighborhood
    radiuscooling='linear'      # Function that defines the decrease in the neighborhood size as the training proceeds (linear, exponential)
    scalecooling='linear'       # Function that defines the decrease in the learning scale as the training proceeds (linear, exponential)
    scale0=0.1                  # Initial learning rate
    scaleN=0.01                 # Final learning rate
    initialization='random'     # Type of SOM initialization (random, pca)
    gridtype='rectangular'      # Type of SOM grid (hexagonal, rectangular)
    #xmlfile="none"              # SOM inputs as an xml file

    kmeans="true"           # Run k-means clustering
    kmeans_init= 5           # Number of initializations
    kmeans_min= 2            # Minimum number of k-mean clusters
    kmeans_max= 25           # Maximum number of k-mean clusters

    output_file_geospace=None   # Text file that will contain calculated values: {X Y Z} data1 data2 dataN som_x som_y cluster b_data1 b_data2 b_dataN in geospace.
    geotiff_input=None      # geotiff_input
    normalized="false"      # Whether the data has been normalized or not
    minN=0                  # Minimum value for normalization
    maxN=1                  # Maximum value for normalization
    label=None              # Whether data contains label column, true or false