import os
import glob

class Args:
    input_file: str            # Input file(*.lrn or list, separated with ",")
    output_folder: str         # Folder to save som dictionary and cluster dictionary
    output_file_somspace: str   # Text file that will contain calculated values: som_x som_y b_data1 b_data2 b_dataN umatrix cluster in geospace
    outgeofile: str             # file name of the geospace output

    #Parameter required for som calculation. 

    som_x = 30                # X dimension of generated SOM
    som_y = 30                # Y dimension of generated SOM
    epochs = 10               # Number of epochs to run

    maptype='toroid'            # Type of SOM ("sheet", "toroid")
    initialcodebook=None        # File path of initial codebook, 2D numpy.array of float32.
    neighborhood='gaussian'     # Shape of the neighborhood function. gaussian or bubble
    std_coeff=0.5               # Coefficient in the Gaussian neighborhood function
    radius0=0.0                   # Initial size of the neighborhood
    radiusN=1.0                   # Final size of the neighborhood
    radiuscooling='linear'      # Function that defines the decrease in the neighborhood size as the training proceeds ("linear", "exponential")
    scalecooling='linear'       # Function that defines the decrease in the learning scale as the training proceeds ("linear", "exponential")
    scale0=0.1                  # Initial learning rate
    scaleN=0.01                 # Final learning rate
    initialization='random'     # Type of SOM initialization ("random", "pca")
    gridtype='rectangular'      # Type of SOM grid ("hexagonal", "rectangular")

    kmeans="true"            # Run k-means clustering
    kmeans_init= 5           # Number of initializations
    kmeans_min= 2            # Minimum number of k-mean clusters
    kmeans_max= 25           # Maximum number of k-mean clusters

    # Additional optional parameter:

    output_file_geospace=None   # Text file that will contain calculated values: {X Y Z} data1 data2 ... dataN som_x som_y cluster b_data1 b_data2 b_dataN in geospace.
    geotiff_input=None        # geotiff_input files, separated by komma, to write GeoTIF out (only first line is used to get the geotransform and projection information to set output GeoFIT geotransform and projection)
    normalized="false"      # Whether the data has been normalized or not ("false", "true")
    #minN=0                  # Minimum value for normalization
    #maxN=1                  # Maximum value for normalization
    label=None              # Whether data contains label column, true or false
    label_geotiff_file = None       # geotiff_input file

    def __init__(self):
        """Constructor for the class
        """

    def create_list_from_pattern(self, file_path, file_patterns):
        """Create a list of full file names from files matching the given file_patterns within the file_path
        :param file_path: The name of the file path for files to be added to the input list.
        :type filename: str.
        :param file_patterns: One or more patterns for file names to be added to the input list.
        :type filename: str.
        :rtype: list of strings
        """

        #-- Lists to store matching files with their corresponding destination paths
        input_list_text = []

        for file_pattern in file_patterns:
            #-- Use glob to get all files with the specified pattern
            matching_files = glob.glob(os.path.join(file_path, file_pattern))

            #-- Add matching files to the list
            input_list_text.extend(matching_files)

        print("Number of files added:", len(input_list_text))
        print("Files:")
        for i in range(min(10, len(input_list_text))): #-- option: change the max number of files to be printed
            print(input_list_text[i])
        if len(input_list_text) > 10:
            print("...")
            print(input_list_text[-1])

        input_file= ",".join(input_list_text)

        return input_file