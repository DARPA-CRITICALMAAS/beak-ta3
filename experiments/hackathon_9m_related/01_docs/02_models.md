# 1 Methods
## 1.1 Introduction 

The following document briefly descibes how to prepare data and run models with given examples.

We provide three machine learning based methods to support critical mineral assessment:
- Self-organizing maps (**SOM**)
- Artificial neural networks (**ANN**)
- Bayesian neural networks (**BNN**) which are still under development 🚧 

Please note that **we do not provide a single model** but a workflow including the creation of models given a specific method. Due to the nature of these methods, each run (of the same example) will most likely result in a different but similar output.

To run a specific method for a selected mineral system, open the respective **notebooks** and follow the instructions. Model names are data and time tagged to avoid accidental overwriting of results.

For each model configuration (e.g. baseline), a single notebook is provided. 

**Results** will be stored into subfolders. 

For **ANNs**:
- Model file
- Logs
- Maps

For **BNNs**:
- Prediction map
- Prediction uncertainty map

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

With this release, we provide the preliminary results to the hackathon 9 month event:
- Porphyry copper (**POCO**) on national scale for the lower U.S. 48 states
- Cobalt nickel (**NICO**) for the U.S. upper midwest region
- Tungsten skarn (**TUSK**) for the Alaska Yokon (YTU) region

For all of these CMAs, results for 
- ANN
- BNN
- SOM
are provided, stating that these are preliminary versions and need to be improved.

The respective model configurations are **all** based on the USGS-provided evidence layers, dated by May 3rd 2024. These setups are named **Feature Fox** models. Additionally, for **POCO** and **TUSK**, two versions, containing only gephysical layers, were build. These models configs are named **Baseline Bison**. 

The model configurations can be investigated by viewing the respective model configuration file, stored in **src/beak/models/**. Alternatively, all of the incorporated layers were saved in the respective model subfolder in **/03_cma/**. 

Regarding the labels, all neural-network based models have been trainined with available mineral occurences **within** a specific area. I.e., the regional models were trained with **only** those points, which are located within the focussed region.

## 1.4 Special models

### Cobalt Nickel Upper Midwest

For both, **ANN** and **BNN**, an additional version was created, that incorporates all available training points in the conterminous U.S. instead of only those within the specific region footprint. These models contain the suffix "**NAT**" or "**nat**" and only apply for the **Feature Fox** configuration.

### Tungsten Skarn Alaska

For both, **ANN** and **BNN**, an additional version was created with the available labels **buffered** by one pixel for comparison with simple random oversampling of training data. These models contain the suffix "**extended**" or "**ext**".

## 1.5 Metrics overview

Metrics are based on evaluation of the **positive** labels only (**not** AUC).

**ANN** results are restricted to an interval \[0,1].  However, **BNN** results can **exceed** these boundaries to a certain degree at the current stage of development. I.e., metrics for BNN results may be overrated at the moment.

| CMA  | Model       | Method | Accuracy | F1 Score | AUC Score* | Num Labels |
| ---- | ----------- | ------ | -------- | -------- | ---------- | ---------- |
| POCO | Bison       | ANN    | 0.740    | 0.851    | 0.994      | 135        |
| POCO | Fox         | ANN    | 0.757    | 0.862    | 0.993      | 99         |
| NICO | Fox         | ANN    | 0.844    | 0.915    | 0.996      | 45         |
| NICO | Fox (nat)   | ANN    | 0.86     | 0.924    | 0.996      | 150        |
| TUSK | Bison       | ANN    | 0.869    | 0.930    | 0.990      | 23         |
| TUSK | Fox         | ANN    | 0.833    | 0.909    | 0.988      | 18         |
| TUSK | Bison (ext) | ANN    | 1.0      | 1.0      | 0.993      | 207        |
| TUSK | Fox (ext)   | ANN    | 1.0      | 1.0      | 0.994      | 171        |
| POCO | Fox         | BNN**  | 0.969    | 0.984    | 0.980      | 99         |
| POCO | Bison       | BNN**  | 0.940    | 0.969    | 0.988      | 135        |
| NICO | Fox         | BNN**  | 0.977    | 0.988    | 0.977      | 45         |
| NICO | Fox (nat)   | BNN**  | 0.986    | 0.993    | 0.986      | 150        |
| TUSK | Bison       | BNN**  | 1.0      | 1.0      | 0.995      | 23         |
| TUSK | Fox         | BNN**  | 1.0      | 1.0      | 0.994      | 18         |
| TUSK | Bison (ext) | BNN**  | 1.0      | 1.0      | 0.995      | 207        |
| TUSK | Fox (ext)   | BNN**  | 1.0      | 1.0      | 0.996      | 171        |
\* incorporates also negative values used for training
\** pixels greater or equal than **0.5** treated as predicted positive
