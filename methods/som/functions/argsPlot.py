class Args:
#-- Arguments for drawing plots from self organizing maps

    outsomfile: str     # dest="outsomfile", help='Som calculation somspace output text file')
    som_x: int          # som x dimension
    som_y:int           # som y dimension
    input_file: str     # Input file(*.lrn)
    dir: str            # Input file(*.lrn)
    grid_type='rectangular' # grid type (rectangular or hexagonal)
    redraw='true'       # whether to draw all plots, or only those required for clustering (true: draw all. false:draw only for clustering).
    outgeofile=None     # som geospace results txt file
    dataType=None       # Data type (scatter or grid)
    noDataValue='NA'    # noData value