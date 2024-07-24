# 1 Introduction
## Overview

With this release @07/23/2024, we provide the following **pre-liminary** hackathon 12 month CMA results for both self-organizing maps and bayesian neural network 🚧 approaches:

- Regional **magmatic nickel and cobalt** in the upper midwest U.S. (**NICO UPW**)
- Regional **mississippi-valley-type** in the central-eastern U.S. area (**MVT CEUS**)
- Regional **porphyry copper** in the southwest U.S (**POCO SW**)

CMAs. Additionally, we run two separated areas for **MVT** and **POCO** applying the **self-organizing maps** only, which include **high-resolution** geophysical data:

- Regional **mississippi-valley-type** in the southern continental U.S. (**MVT SMC**)
- Regional **porphyry copper** in southwest New Mexico (**POCO SWNM**)

For these models, the lower-resolution data were **substituted** with their high-resolution counterparts (pixel size **50 m** instead of **500** m). Other data needed to run the models were artificially resampled to fit the smaller pixel size.

Due to the nature of these methods, each run (of the same example) will most likely result in a different but similar output. 

⚠  Please note that the **bayesian neural network** based models are **not yet** in **[0, 1]** range, which is still **ongoing** development and we aim to **fix** this ASAP. Due to this circumstance, there will be no metrics available since the classification measures would be not reliable in this case.

⚠ For the **self-organizing maps**, we also provide a label correlation (if those were available) based on the resulting **BMU** and **k-means** clusters, which is basically the number of positive labels that is located within a certain unit. It is an **optional** step **after** the creation of the **SOM** results and can **support the identification of potentially prospective areas** or used to **sample additional points for a supervised approach**.

## Results

We provide follwing results for the **SOM** and **BNN** models:

For **BNNs**
- Prediction map
- Prediction uncertainty map

For **SOM**
- K-Means cluster results (**cluster**)
- Best matching units (**BMU**) with associated label count (**bmu_bmu_label_count**)
- BMU with associated label count of the k-means cluster (**bmu_cluster_label_count**)
- Q-Error
- **Plots**
- Logs

# 2 Data

All necessary results are part of a **QGIS** project. For each **CMA**, the **plots** and logs and configs etc. are located within the **CMA** folder.

Due to the three CMAs, different methods and models, the map export became quite large. For that reason, and because of some errors in the "all-in-one" export, we provide three separate files/folders/projects for **download**:

- Data (**preprocessed** data used for model input)
- Regional (**three regional** CMAs)
- Regional High-Res (**two regional** CMAs with **high-res** data)

Each folder contains a **QGIS** project and associated data, containing only **selected** model runs. 

Within the GIS project, the **preferred** models were marked with an asterix' "\*". The **more** **"\*"**, the more reliable the results (biased, subjective interpretation). For comparison and sake of completeness, we kept all initial **BISON** and **JELLYFISH** and some other model runs, too. These were not marked in any way.

⚠ The respective data used for modeling are stored in the **data** folder. If needed copy/paste them from one to the other project.

⚠ For **comparison**, we added selected results from the **pre-hack** and **hack 9 month** to the project.

# 3 Models

The respective model configurations are **all** based on the USGS provided evidence layers/shared sheet documet, from end of week 29. 

Like in hackathon 9, we provide model runs for the 100% (or most comparable) match-up from USGS in the shared CMA document. These models are called **JITTER JELLYFISH**. Additionally, for each of these models, a **geophysical baseline** model was created, incorporating only deeper geophysical layers (all surface-related evidence layers like geology, surface geophysics etc. were excluded). These are called **BASELINE BISON**. 

Regarding the labels, all neural-network based models have been trainined with available mineral occurences **within** a specific area. I.e., the regional models were trained with **only** those points, which are located within the focussed region.

Since the **high density** of points for

- Shallow magnetic worms
- Shallow gravity worms
- Isostatic gravity worms

seemed to cause unreliable model results, especially for the **SOM** clusters, we decided to remove them from the model configuration and run each of our models without these (depending on whether they were part of the config or not). The new models are called **DROPOUT DUCK** and **LOSS LLAMA**, related to the **BISON** and **JELLYFISH** models, respectively.

For **MVT** **CEUS**, we also added the **FEATURE FOX** configuration from hackathon 9 month and run it with the new labels (12 month). For comparison, older models from the previous hackathon were added to the QGIS project and data export as well.

In addition, the high-resolution CMAs for **MVT SMC** and **POCO SWNM** were run with both the standard configuration that only includes the **resampled** data (no native high-res) and a version which **contains the high-res** data (substituting the low-res counterpart). For these, we added **PP** (stands for pixel peeper and relates to small pixels) as **suffix** to the model name. 

Combinations for models and methods are listed below.

**Standard** models for **SOM** and **BNN**:

|           | **MVT** | **NICO** | **POCO** |
| --------- | ------- | -------- | -------- |
| BISON     | x       | x        | x        |
| JELLYFISH | x       | x        | x        |
| DUCK      | x       | x        | x        |
| LLAMA     | x       | x        | x        |
| FOX       | x       | -        | -        |
| GOPHER    | x       | -        | x        |
| YAK       | x       | -        | x        |
| CAMEL     | -       | -        | x        |
| PENGUIN   | -       | -        | x        |

**High-res** models for SOM only:

|                  | **MVT SMC** | **POCO SWNM** |
| ---------------- | ----------- | ------------- |
| BISON            | x           | x             |
| **BISON PP**     | x           | x             |
| JELLYFISH        | x           | x             |
| **JELLYFISH PP** | x           | x             |
| DUCK             | x           | x             |
| **DUCK PP**      | x           | x             |
| LLAMA            | x           | x             |
| **LLAMA_PP**     | x           | x             |

