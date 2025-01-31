
# Dropout Evaluation
05-09-2024
## Porphyry Copper
### Explanation

With this release, we provide the preliminary results for the evaluation with a complete label model (100%) and a 20% dropout model (80%) for Porphyry copper (**POCO**) on national scale for the lower U.S. 48 states. 

The model configurations can be investigated by viewing the respective model configuration file, stored in **src/beak/models/**.

There are two model configurations tested:
1. Baseline Bison
2. Feature Fox

The Fox models represent the suggested configuration provided by USGS prior to the 9-month hackathon. The Bison models represent a smaller configuration with geophysical layers only (e.g., no geochemistry). 

Models were build for ANN and BNN methods. Additionally, **.shp** and **.csv** files were created with the prediction results for ANN (value) and BNN (value, uncertainty). These are located in the **/05_evaluation/data** folder and are also part of the QGIS project located in **/04_qgis/**.

The provided shapefile/csv contains the extracted results from each model run. 
The respective columns are named like: **A_BIS_80** or **B_FOX_U**

- **A** refers to ANN and **B** to BNN
- **BIS** to Bison and FOX **to** Fox model configurations (evidence layers)
- **80** corresponds to the 20% dropout labels used for training provided by MITRE/USGS
- **U** represents the uncertainty output of the BNN

The **SEL_80** column stores the information whether the label was part of the 20% dropout file (0: no, 1: yes).

Since the **FOX** model has a much lower data coverage, a certain number of points from both the 100% and the 80% labels were not used for training due to the lack of data (30 for the complete, 10 for the dropout version). These cells/pixels are empty (no prediction for those) and were marked with values of **-9999**.

### Metrics overview

Metrics are based on evaluation of the **positive** labels only (**not** AUC).

**ANN** results are restricted to an interval \[0,1].  However, **BNN** results can **exceed** these boundaries to a certain degree at the current stage of development. I.e., metrics for BNN results may be overrated at the moment.

**Bold number of labels** refers to the 80% version.

| CMA  | Model | Method | Accuracy | F1 Score | AUC Score* | Num Labels |
| ---- | ----- | ------ | -------- | -------- | ---------- | ---------- |
| POCO | Bison | ANN    | 0.740    | 0.851    | 0.994      | 135        |
| POCO | Fox   | ANN    | 0.757    | 0.862    | 0.993      | 99         |
| POCO | Bison | BNN**  | 0.940    | 0.969    | 0.988      | 135        |
| POCO | Fox   | BNN**  | 0.969    | 0.984    | 0.980      | 99         |
| POCO | Bison | ANN    | 0.745    | 0.854    | 0.989      | **110**    |
| POCO | Fox   | ANN    | 0.647    | 0.696    | 0.987      | **85**     |
| POCO | Bison | BNN**  | 1.0      | 1.0      | 0.995      | **110**    |
| POCO | Fox   | BNN**  | 1.0      | 1.0      | 0.999      | **85**     |
\* incorporates also negative values used for validation
\** pixels greater or equal than **0.5** treated as predicted positive

