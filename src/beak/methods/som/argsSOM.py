import os
import glob
import json

class Args:
    """Arguments for SOM and k-means clustering.

    Args:
        input_file (str): Input file(".lrn " or list, separated with ",").
        output_folder (str): Folder to save SOM dictionary and cluster dictionary.
        output_file_somspace (str): Text file that will contain calculated values.
            Parameters: som_x som_y b_data1 b_data2 b_dataN umatrix cluster in geospace.
        outgeofile (str): File name of the geospace output.
        
        som_x (int): X dimension of generated SOM.
        som_y (int): Y dimension of generated SOM.
        epochs (int): Number of epochs to run.
        maptype (str): Type of SOM ("sheet", "toroid").
        initialcodebook (Optional[str]): File path of initial codebook,
            2D numpy array of float32.
        neighborhood (str): Shape of the neighborhood function. gaussian or bubble.
        std_coeff (float): Coefficient in the Gaussian neighborhood function.
        radius0 (float): Initial size of the neighborhood.
        radiusN (float): Final size of the neighborhood.
        radiuscooling (str): Function that defines the decrease in the neighborhood 
            size as the training proceeds ("linear", "exponential").
        scalecooling (str): Function that defines the decrease in the learning scale 
            as the training proceeds ("linear", "exponential").
        scale0 (float): Initial learning rate.
        scaleN (float): Final learning rate.
        initialization (str): Type of SOM initialization ("random", "pca").
        gridtype (str): Type of SOM grid ("hexagonal", "rectangular").
        
        kmeans (str): Run k-means clustering ("true" or "false").
        kmeans_init (int): Number of initializations.
        kmeans_min (int): Minimum number of k-mean clusters.
        kmeans_max (int): Maximum number of k-mean clusters.
        
        output_file_geospace (Optional[str]): Text file that will contain calculated values.
            Parameters: {X Y Z} data1 data2 ... dataN som_x som_y cluster b_data1 b_data2 b_dataN in geospace.
        geotiff_input (Optional[str]): Geotiff_input files, separated by comma, to write GeoTIF out 
            (only first line is used to get the geotransform and projection information to 
            set output GeoFIT geotransform and projection)
        normalized (str): Whether the data has been normalized or not ("true" or "false").
        label (Optional[str]): Whether data contains label column ("true" or "false").
        label_geotiff_file (Optional[str]): Geotiff_input file.
    """    

    def __init__(self):
        """Constructor for the class. Initialize all attributes with default values
        """
        self.input_file = ""
        self.output_folder = ""
        self.output_file_somspace = ""
        #self.outgeofile = ""
        self.som_x = 30
        self.som_y = 30
        self.epochs = 10
        self.maptype = 'toroid'
        self.initialcodebook = None
        self.neighborhood = 'gaussian'
        self.std_coeff = 0.5
        self.radius0 = 0.0
        self.radiusN = 1.0
        self.radiuscooling = 'linear'
        self.scalecooling = 'linear'
        self.scale0 = 0.1
        self.scaleN = 0.01
        self.initialization = 'random'
        self.gridtype = 'rectangular'
        self.kmeans = True
        self.kmeans_init = 5
        self.kmeans_min = 2
        self.kmeans_max = 25
        self.output_file_geospace = None
        self.geotiff_input = False
        self.normalized = "false"
        self.label = False
        self.label_geotiff_file = None

    @classmethod
    def from_json_file(cls, file_path):
        """Creates an instance of Args from a JSON file.

        Args:
            file_path (str): The path to the JSON file containing arguments data.

        Returns:
            Args: An instance of Args populated with data from the JSON file.
        """
        with open(str(file_path), 'r') as file:
            data = json.load(file)

        args_instance = cls()
        args_instance.__dict__.update(data)  # Update instance attributes with JSON data
        return args_instance  

    def create_list_from_pattern(self, file_path, file_patterns):
        """Create a list of full file names from files matching the given file_patterns within the file_path

        Args:
            file_path (str): The name of the file path for files to be added to the input list.
            file_patterns (str): One or more patterns for file names to be added to the input list.

        Returns:
            LiteralString: list of full file names, separated by ","
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
    
    def create_list_from_file(self, file_path, file_path_label=""):
        """create a list of input file names and path

        Args:
            file_path (str): file path to TXT file holding a list of GeoTIF file paths
            file_path_label (str, optional): file path to TXT file holding a list of label data. Defaults to "".

        Returns:
            LiteralString: a list of full file names from files matching the given files in txt file. Separated by ",".
        """
        # Check if label file exists
        if os.path.exists(file_path_label):
            # Open the label file in read mode
            with open(file_path_label, 'r') as label_file:
                # Read the whole content from the label file
                content_label = label_file.readlines()
        else:
            content_label = []  # If the label file doesn't exist, initialize an empty list

        # Open the list file in read mode
        with open(file_path, 'r') as file:
            # Read all lines from the file into a list
            content = file.readlines()

        # Extend content with content_label if it exists
        content.extend(content_label)

        # Strip newline characters from each line
        file_patterns = [line.strip() for line in content]

        input_file = self.create_list_from_pattern("", file_patterns)

        return input_file
    
