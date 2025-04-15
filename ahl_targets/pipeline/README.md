This folder contains scripts that generate data used further downstream in the project.

- `coefficients.py` runs linear regressions of energy density and NPM scores and saves coefficient tables
- `model_data.py` generates the source data for the main model
- `npm_robustness.py` generates the source data for the robustness checks
- `simulation_ed.py` energy density model
- `simulation_hfss.py` HFSS model
- `simulation_npm.py` NPM model

######

# NPM Modelling Update 02/12/2024

## Summary

This folder contains the scripts used to update the input data and methodology used in calculating retailer health targets. Since the development of this model, subsequent work, such as on the "Diets: State of the Nation" (DSotN) report, has enabled us to iterate on and improve the impact modelling. Here we will present an overview of these updates and the impact, followed by more detail on the changes.

## Outcome

### Impact

| Model       | Avg kcal per person per day (baseline) | Avg kcal per person per day (post) | Targets Impact (kcal pp per day reduction) | NPM sales-weighted average (baseline) | NPM sales-weighted average (post) |
| ----------- | -------------------------------------- | ---------------------------------- | ------------------------------------------ | ------------------------------------- | --------------------------------- |
| 2024 Update | 1907                                   | 1833                               | 74                                         | 1.86                                  | 0.79                              |
| Original    | 1629                                   | 1579                               | 50                                         | 1.54                                  | 0.53                              |

The impact of the targets on obesity are then calculated using the average calories consumed per person per day as an input to the [Hall model](https://pubmed.ncbi.nlm.nih.gov/21872751/).

For a more detailed comparison of the model outputs, see `ahl_targets/analysis/2024_update/output_comparison.py`.

## Updates overview

### 1. Input data update

The input data file has been updated to be consistent with the output files from DSotN. This should have no effect as the underlying data is the same, just more consistently processed to make work across projects easier. This file is currently created here in the DSotN repository [here](https://github.com/nestauk/ahl_diets_evidence/blob/45_targets_data/ahl_diets_evidence/pipeline/number_calories_gb_retailer_checks.py).

To do: Update to read in from `ahl_core_data` instead.

### 2. Only consider adult calorie intake

The model is updated to only calculate the impact of the retailer targets model on adult calorie intake. Correspondingly, the mean kcal pp per day is higher, as it isn't dragged down by all of the age groups that consume far fewer calories per day.

This is done using a methodology developed in the DSotN project, whereby the average kcal intake of children is estimated from their age and gender, and therefore a conversion factor for the proportion of a household's intake attributable to adults is calculated. The functions used to calculate can be found in `ahl_targets/utils/diets.py`.

### 3. Increasing the number of categories in scope

Several food categories were excluded from the original modelling as they are measured in litres, so were deemed to be drinks. Several of these have been reintroduced as have been judged to be in scope after reviewing [NPM technical guidance](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/694145/Annex__A_the_2018_review_of_the_UK_nutrient_profiling_model.pdf). These categories are:

- Cooking Oils
- Total Ice Cream
- Fresh Cream
- Lards+Compounds
- Vinegar
- Breakfast Cereals
- Defined Milk+Cream Prd(B)

N.B. The category field used in the data is `rst_4_market`.

### 4. Apply specific gravity adjustment

The above food categories have volume measured in litres, and have have varying densities such that the water density approximation (1L ~ 1kg) is insufficiently accurate. As a result we need to apply a "specific gravity" adjustment. Currently this adjustment is applied at the `rst_4_market_sector` category level and uses values set by the Food Standards Agency (in [this book](https://www.tsoshop.co.uk/bookstore.asp?ACTION=BOOK&PRODUCTID=9780112429616))

### 5. Remove data entry errors

There exist some outlying products in the data that are now deemed to be data entry errors. Those removed include those that appear to have an energy density of greater than 900 kcal/100g (impossible) and those with a volume of 0. This is best practice although the impact on the result is negligible: 0.0075% of the total products are removed.

## Detail: How do we arrive at a 1907kcal per person per day baseline, when the DSotN figure is 2393kcal?

The underlying data undergoes a series of transformations to meet the needs of this project.

1. Jan-March is reintroduced. The retailer targets model is run on the whole year, whereas the diets figure is calculated from just the Apr-Dec subset.
2. Stores that are not in scope are removed. The retailer targets model is just run on purchases from the 11 largest retailers, whereas the diets figure is calculated from all in-home purchases.
3. Drinks are removed. The retailer targets model is just run on food products.
4. Unphysical products (data quality errors) are removed. These are those with an energy density >900 and those with a volume of 0.

N.B A good way to understand this is to run [this script](https://github.com/nestauk/ahl_diets_evidence/blob/45_targets_data/ahl_diets_evidence/pipeline/number_calories_gb_retailer_checks.py) in the diets repository which performs the transformation.

## Detail: Explaining the difference in the impact of the model

The most surprising aspect of the update is the increase in the relative reduction in average kcal per person per day. Where the original model predicted a 50kcal reduction due to the modelling parameters set in `ahl_targets/config/npm_model.yaml`, the new model predicts a 74 kcal reduction.

This new larger figure is investigated in `ahl_targets/analysis/2024_update/output_comparison.py` which suggests it is a combination of two factors:

1. (Most impactful) As the reincluded products are typically HFSS (ice creams/oils), a greater proportion of products are selected for reformulation and negative sales shifts. While fewer products get selected for positive sales shifts (non-HFSS products), these increases (9%) are smaller than the decreases (10.5%) given the set parameters.
2. As the reincluded products are typically high-NPM and high energy density, the regression coefficients in the categories including these products increase. In the case of the 'Savoury Home Cooking` category, the re-introduction of cooking oils takes the coefficient from 6.3 to 16.9. Therefore, each instance of reformulation has a much greater modelled effect on calorie density.
