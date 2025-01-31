# 1 Tools

## 1.1 Introduction

We provide different tools for 
- data preprocessing
- creation of models for critical mineral assessment

The preprocessing examples are meant to **showcase** how the preprocessing was accomplished, but it is not necessary to run them, since all data are contained in the provided package. 

To start modeling, all model input data **must have** the **same**
- coordinate reference system
- spatial resolution
- spatial extent

Examples within
- `../02_data_preparation` contain workflow for a specific dataset
	- conversion from datacube to raster data (only Lawley datacube)
	- spatial alignment to a predefined template raster (so called **base raster**), including
		- resampling
		- reprojection
		- clipping
		- masking
	- Imputation
	- Transformation/scaling

- `/examples/notebooks` contains different single examples
	- creation of a base raster or template based on a shapefile with queries
	- creation of labels (raster) to be used for model input
	- snapping raster to another raster's origin
	- "all-in-one" unification raster data based on a provided base raster

Imputation and transformation of data is only needed for the **SOM** approach. The examples provided for the neural networks cover these steps within the notebooks.

If there are any open questions, please check out the [code documentation](../../../docs/build/html/index.html).

## 1.2 Examples

Examples are generally provided in the following structure:
**dataset_name`/`EPSG_number_RES_pixelsize`/`Spatial_Extent**

E.g., `../dataset_lawley22_datacube/EPSG_4326_RES_0_025/US_CANADA`  means that the notebooks contained in that folder will prepare the data based on the `dataset_lawley22_datacube` and convert them into `EPSG_4326` with a resolution `RES_0_025` degree for the `US_Canada` territory.

### Lawley (2022) datacube

The corresponding notebooks for converting the datacube into a raster format with specific
- spatial extent
- spatial resolution
- coordinate reference system (**CRS**)
 
are located in `../02_data_preparation/dataset_lawley22_datacube/EPSG_4326_RES_0_025/US_CANADA`

The data are stored with the EPSG-code 4326, 0.025 degree pixel-size and cover the U.S. and Canada. The other subfolders
- `US_ALASKA` (only Alaska)
- `US_CONT` (only U.S. lower 48 states)

are clipped versions and based on this export. The version `EPSG_3857_RES_2000` is an example how to convert all initially exported layers into another **CRS** with metric units (in this case, 2.000 pixel size).

### Other datasets

The other examples provided basically convert and unify **RAW** raster data for a selected dataset into the needed specifications for modeling based on a **template** raster that provides the necessary metadata for the spatial transformation.

## 1.3 Running preprocessing

Choose a dataset for preprocessing and run the notebooks contained within the folder. If those, follow additional instructions provided in the notebooks.

The **output paths** are set to be created within the notebooks folder, so that no already existing data will be overwritten. We provided all necessary data to run the models in the release.


## 1.2 Requirements

All tools are designed to run on lower-end machines, e.g. consumer PCs and laptops with 16 GB of memory and processing units with low number of cores. However, some tools use multi-processing by default, which can cause memory-overflow. In that case, lower the number of cores to be used (e.g. to 1).

The process of raserization of the Lawley datacube is very time consuming and computational expensive. We recomment **not** to run these on multiple cores unless there are at least 128 GB of free memory available in your system. The export on **numerical** data for this set took about 17 minutes. With all data (numerical, categorical, grount truth), close to 4 hours by utilizing 48 threads on a machine with 256 GB of RAM.