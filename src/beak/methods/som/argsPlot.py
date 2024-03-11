class Args:  
    """Arguments for drawing plots from SOM results.

    Args:
        outsomfile (str): SOM calculation somspace output text file.
        som_x (int): SOM x dimension.
        som_y (int): SOM y dimension.
        input_file (str): Input file (".lrn").
        dir (str): Input file (".lrn").
        grid_type (str): Grid type ("rectangular" or "hexagonal").
        redraw (bool): Whether to draw all plots, or only those required for clustering.
            "True": draw all. "False": draw only for clustering.
        outgeofile (str): SOM geospace results txt file.
        dataType (str): Data type ("scatter" or "grid").
        noDataValue (str): Nodata value.
    """    
    outsomfile: str         # SOM calculation somspace output text file
    som_x: int              # SOM x dimension
    som_y:int               # SOM y dimension
    input_file: str         # Input file(*.lrn)
    dir: str                # Input file(*.lrn)
    grid_type='rectangular' # Grid type (rectangular or hexagonal)
    redraw=True             # Whether to draw all plots, or only those required for clustering (True: draw all. False:draw only for clustering).
    outgeofile=None         # SOM geospace results txt file
    dataType=None           # Data type (scatter or grid)
    noDataValue='NA'        # Nodata value