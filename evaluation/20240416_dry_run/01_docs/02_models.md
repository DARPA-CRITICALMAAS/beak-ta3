# 1 Methods
## 1.1 Introduction 

The following document briefly descibes how to prepare data and run models with given examples.

We provide three machine learning based methods to support critical mineral assessment:
- Self-organizing maps (**SOM**)
- Artificial neural networks (**ANN**)
- Bayesian neural networks (**BNN**) which are still under development 🚧 

Please note that **we do not provide a single model** but a workflow including the creation of models given a specific method. Due to the nature of these methods, each run (of the same example) will most likely result in a different but similar output.

To run a specific method for a selected mineral system, open the respective notebooks located in
**beak-ta3/evaluation/critical_mineral_assessements** and follow the instructions. The model descriptions are date and time tagged, so that no existing results will be overwritten.

For each model configuration (e.g. baseline), a single notebook is provided. 

**Results** will be stored into subfolders. 

For **ANNs**:
- Model file
- Logs
- Maps

For **SOM**:
- Cluster results (k-means)
- Best matching units (**BMU**) with associated label count (if labels were provided)
- BMU with associated label count of the k-means cluster (if labels were provided)
- Q-Error
- Plots
- Logs

## 1.2 Requirements

Below an overview for recommended requirements for training and running the models. You can try to run the notebooks below these specifications, but it may more likely result in longer computational time or memory issues. Requirements may also **change** dramatically depending on **scale** and **number of evidence layers** to be used. Given numbers are related to **baseline** models with fewer (12-18) numerical evidence layers.

Since the provided **Docker** container does **not** offer GPU-support, all computations are **CPU-based**. GPU is currently only supported on appropriately prepared native **Linux** systems or Windows **WSL2** in combination with the conda install approach (for neural networks). The **SOM** automatically uses available **GPU** resources to a certain degree due to direct integration of CUDA capabilities.

*Consumer* is related to AMD **Ryzen** or Intel **I**x series processors (or comparable) of the last 4 years. *Workstation* refers to processing units with multiple cores (16 +), capable of supporting 128 + GB memory such like AMD **Threadripper** or Intel **Xeon** series.

Please keep in mind that running these in a **Docker** or **WSL2** environment may lead to **higher** memory **consumption** compared to native Windows or Linux installations.

| Method | Scale         | Pixel size  | RAM [GB] | CPU         | Storage for results [GB] |
| ------ | ------------- | ----------- | -------- | ----------- | ------------------------ |
| SOM    | U.S.          | ~ 6.25 sqkm | 16 +     | consumer    | 1 - 5                    |
| SOM    | U.S. + Canada | ~ 6.25 sqkm | 64 +     | consumer    | unknown                  |
| SOM    | regional      | 50 m        | 128 +    | workstation | **40** +                 |
| xNN    | U.S.          | ~ 6.25 sqkm | 16 +     | consumer    | < 1                      |
| xNN    | U.S. + Canada | ~ 6.25 sqkm | 16 +     | consumer    | < 1                      |
| xNN    | regional      | 50 m        | unknown  | unkown      | < 1                      |

## 1.3 Model overview
With this release, we cover the mineral systems:
- Mississippi-Valley-Type deposits (**MVT**) on national scale
  *Configurations for these models are based on the Lawley et al. (2022) paper
  
- **Magmatic-Nickel** on national scale
  *Configurations are based on the outcomes of the 6 months hackathon*

| CMA                         | SOM | ANN | BNN                                   | Scale             | Configurations                     |
| --------------------------- | --- | --- | ------------------------------------- | ----------------- | ---------------------------------- |
| Mississippi-<br>Valley-Type | -   | x   | -                                     | U.S. + Canada     | Baseline<br>Preferred              |
| Mississippi-<br>Valley-Type | x   | -   | -                                     | conterminous U.S. | Baseline<br>Preferred + Isogravity |
| Mississippi-<br>Valley-Type | -   | x   | -                                     | conterminous U.S. | Preferred + Isogravity             |
| Magmatic <br>Nickel         | x   | x   | exemplary<br>**intermediate** results | conterminous U.S. | Baseline<br>                       |

## 1.4 Evidence layers

All necessary data needed to run tools for data preprocessing or creating models for critical mineral assessment are contained within the `/beak-ta3/src/beak/data` folder.

All model configurations are saved within a respective model file for each CMA:
`/beak-ta3/src/beak/data/models`

Personal communication as source refers to data exchanged during the hackathon 6 month event.

### Mississippi-Valley-Type, **Baseline**
Configuration key: **MVT_BASELINE**

For the **baseline** model, two versions were built:
1. (**DC**) version, completely based on the **direct** datacube input (not rasterized)
2. An alternative version based on the **rasterized** data from the Lawley et al. (2022) datacube.

The SOM takes only rasterized input data (2), i.e., there is **no** DC version (1) for this approach.

| #   | Evidence Layer                                 | Source        |
| --- | ---------------------------------------------- | ------------- |
| 1   | Gravity_GOCE_ShapeIndex                        | Lawley (2022) |
| 2   | Gravity_Bouguer                                | Lawley (2022) |
| 3   | Gravity_Bouguer_HGM                            | Lawley (2022) |
| 4   | Gravity_Bouguer_UpCont30km_HGM                 | Lawley (2022) |
| 5   | Gravity_Bouguer_HGM_Worms_Proximity            | Lawley (2022) |
| 6   | Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity | Lawley (2022) |
| 7   | Magnetic_HGM                                   | Lawley (2022) |
| 8   | Magnetic_LongWavelength_HGM                    | Lawley (2022) |
| 9   | Magnetic_HGM_Worms_Proximity                   | Lawley (2022) |
| 10  | Magnetic_LongWavelength_HGM_Worms_Proximity    | Lawley (2022) |
| 11  | Seismic_LAB_Hoggard                            | Lawley (2022) |
| 12  | Seismic_Moho                                   | Lawley (2022) |
### Mississippi-Valley-Type, **Preferred**

Configuration keys: **MVT_PREFERRED**, **MVT_PREFERRED_ISOGRAV**

For the **preferred** model, two versions were built:
1. (**DC**) version, completely based on the **direct** datacube input (not rasterized)
2. An alternative version based on the **rasterized** data from the Lawley et al. (2022) datacube

The SOM takes only rasterized input data (2), i.e., there is **no** DC version (1) for this approach.

The isogravity version substitudes the **Gravity_Bouger** and **Gravity_Bouger_HGM** data with **isostatic** **gravity** data and also replaces some of the rasterized layers from the datacube with unified raster data from the McCafferty (2023) dataset. However, this is a 1:1 replacement, the provided **evidence** remains the same! The **isogravity** version is currently restricted to the conterminous U.S. and thus, the number of training points is significantly lower (1.794 vs. 1.352 for the rasterized **EPSG 4326 RES 0025** versions). No **DC** version has been built for this approach.

| #   | Evidence Layer                                      | Classes | MVT_PREFERRED | MVT_PREFERRED_ISOGRAV |
| --- | --------------------------------------------------- | ------- | ------------- | --------------------- |
| 1   | Geology_Lithology_Majority                          | 31      | Lawley (2022) | Lawley (2022)         |
| 2   | Geology_Lithology_Minority                          | 31      | Lawley (2022) | Lawley (2022)         |
| 3   | Geology_Period_Maximum_Majority                     | 20      | Lawley (2022) | Lawley (2022)         |
| 4   | Geology_Period_Minimum_Majority                     | 20      | Lawley (2022) | Lawley (2022)         |
| 5   | Geology_Dictionary_Calcareous                       | 1       | Lawley (2022) | Lawley (2022)         |
| 6   | Geology_Dictionary_Carbonaceous                     | 1       | Lawley (2022) | Lawley (2022)         |
| 7   | Geology_Dictionary_FineClastic                      | 1       | Lawley (2022) | Lawley (2022)         |
| 8   | Geology_Dictionary_Felsic                           | 1       | Lawley (2022) | Lawley (2022)         |
| 9   | Geology_Dictionary_Intermediate                     | 1       | Lawley (2022) | Lawley (2022)         |
| 10  | Geology_Dictionary_UltramaficMafic                  | 1       | Lawley (2022) | Lawley (2022)         |
| 11  | Geology_Dictionary_Anatectic                        | 1       | Lawley (2022) | Lawley (2022)         |
| 12  | Geology_Dictionary_Gneissose                        | 1       | Lawley (2022) | Lawley (2022)         |
| 13  | Geology_Dictionary_Schistose                        | 1       | Lawley (2022) | Lawley (2022)         |
| 14  | Terrane_Proximity                                   | -       | Lawley (2022) | Lawley (2022)         |
| 15  | Geology_PassiveMargin_Proximity                     | -       | Lawley (2022) | Lawley (2022)         |
| 16  | Geology_BlackShale_Proximity                        | -       | Lawley (2022) | Lawley (2022)         |
| 17  | Geology_Fault_Proximity                             | -       | Lawley (2022) | Lawley (2022)         |
| 18  | Geology_Paleolatitude_Period_Maximum                | -       | Lawley (2022) | Lawley (2022)         |
| 19  | Geology_Paleolatitude_Period_Minimum                | -       | Lawley (2022) | Lawley (2022)         |
| 20  | Gravity_GOCE_ShapeIndex                             | -       | Lawley (2022) | McCafferty (2023)     |
| 21  | Gravity_Bouguer / US_IsostaticGravity_WGS84         | -       | Lawley (2022) | McCafferty (2023)     |
| 22  | Gravity_Bouguer_HGM / US_IsostaticGravity_HGM WGS84 | -       | Lawley (2022) | McCafferty (2023)     |
| 23  | Gravity_Bouguer_UpCont30km_HGM                      | -       | Lawley (2022) | McCafferty (2023)     |
| 24  | Gravity_Bouguer_HGM_Worms_Proximity                 | -       | Lawley (2022) | Lawley (2022)         |
| 25  | Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity      | -       | Lawley (2022) | Lawley (2022)         |
| 26  | Magnetic_HGM                                        | -       | Lawley (2022) | McCafferty (2023)     |
| 27  | Magnetic_LongWavelength_HGM (DeepSources)           | -       | Lawley (2022) | McCafferty (2023)     |
| 28  | Magnetic_HGM_Worms_Proximity                        | -       | Lawley (2022) | Lawley (2022)         |
| 29  | Magnetic_Long_Wavelength_HGM_Worms_Proximity        | -       | Lawley (2022) | Lawley (2022)         |
| 30  | Seismic_LAB                                         | -       | Lawley (2022) | McCafferty (2023)     |
| 31  | Seismic_Moho                                        | -       | Lawley (2022) | McCafferty (2023)     |

### Magmatic Nickel, Baseline

Combines the **source** and **pathway** evidence layers figured out during the geology sessions at the 6 months hackathon event with USGS experts.

| #   | Evidence Layer                                 | Source                 |
| --- | ---------------------------------------------- | ---------------------- |
| 1   | US_IsostaticGravity_WGS84                      | McCafferty (2023)      |
| 2   | Gravity_Up30km_HGM                             | McCafferty (2023)      |
| 3   | Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity | Lawley (2022)          |
| 4   | CONUS_MT2023_9km_cog                           | Murphy (2023)          |
| 5   | CONUS_MT2023_15km_cog                          | Murphy (2023)          |
| 6   | CONUS_MT2023_30km_cog                          | Murphy (2023)          |
| 7   | MagRTP                                         | McCafferty (2023)      |
| 8   | MagRTP_HGMDeepSources                          | McCafferty (2023)      |
| 9   | Magnetic_LongWavelength_HGM_Worms_Proximity    | Lawley (2022)          |
| 10  | Mag_AnalyticSignal_cog                         | McCafferty, pers. com. |
| 11  | Moho                                           | McCafferty (2023)      |
| 12  | LAB                                            | McCafferty (2023)      |
| 13  | LAB_HGM_cog                                    | McCafferty, pers. com. |
| 14  | LAB_Worms_Proximity                            | McCafferty (2023)      |
