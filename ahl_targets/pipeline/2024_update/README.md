UPDATES IN PROGRESS (27/11/24) - README TO BE UPDATED!

In October 2024 we updated the targets model used to produce retailer targets numbers.

The update changed the input data to:

- [FOR NOW] Read in data from the ahl_diets_evidence repo, created in this script: <ahl_diets_evidence/pipeline/number_calories_gb_retailer_checks.py>
- [TBC FUTURE WORK] Read in data from ahl_core_data, including updated NPM scores (with differences for <0.25% to original scores)
- Adjust the kcal and volume values to reflect the adult proportions of household intake created in the diets work
- Add in categories excluded in the retailer targets original work as they were measured in litres, but judged as in scope when reviewing NPM technical guidance (https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/694145/Annex__A_the_2018_review_of_the_UK_nutrient_profiling_model.pdf). The categories added back in are defined rst_4_market categories "Cooking Oils", "Total Ice Cream", "Fresh Cream", "Lards+Compounds", "Vinegar", "Breakfast Cereals", "Defined Milk+Cream Prd(B)".
- Adjust the volume of the added categories to create equivalent measures between kg and litres
- Remove 35176 products which were not included in the original targets work and are not in the categories originally added back in. Of these, 35130 were missing NPM values and the remaining 46 account for just 0.82kcal pp per day. Analysis of this and creation of the list of products to remove are saved in <ahl_targets/ahl_targets/analysis/change_checks_Oct24/compare_new_to_old_file.py>

These changes are implemented in <ahl_targets/ahl_targets/pipeline/Oct_24_updates/create_new_datafile.py>

[TBC] We also made minor adjustments to the model to:

[TBC] This resulted in changes to both the population total kcal reduction from the model, and the SWA NPM baseline of each store. Updated charts are saved in:

###############################################

Updated Readme in progress below:

# NPM Modelling Update 02/12/2024

All of these changes are implemented in [].

## Summary

This folder contains the scripts used to update the input data and methodology used in calculating retailer health targets. Since the development of this model, subsequent work, such as on the "Diets: State of the Nation" (DSotN) report, has enabled us to iterate on and improve the impact modelling. Here we will present an overview of these updates and the impact, followed by more detail on the changes.

## Outcome

### Impact

On kcal_pp_per_day
On swa_npm shifts by store

Include figures

## Updates overview

### 1. Input data update

The input data file has been updated to be consistent with the output DSotN. This should have no effect as essentially the data is the same, just much better and consistently processed to make work across projects easier. This file is currently created here [to add]

[To do: Update to read in from `ahl_core_data` instead]

### 2. Only consider adult calorie intake

The model is updated to only calculate the impact of the retailer targets model on adult calorie intake. Accordingly, the mean is more meaningful, as it isn't dragged down by all of the age groups that consume far fewer calories per day.

This is done using a methodology developed in the DSotN project, whereby the average kcal intake of children is estimated from their age and gender, and therefore a conversion factor for the proportion of a household's intake attributable to adults is calculated. The functions used to calculate can be found in `ahl_targets/utils/diets.py`.

### 3. Increasing the number of categories in scope

Several food categories were excluded from the original modelling as they are measured in litres, but are judged to be in scope after reviewing [NPM technical guidance](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/694145/Annex__A_the_2018_review_of_the_UK_nutrient_profiling_model.pdf). These categories are:

- Cooking Oils
- Total Ice Cream
- Fresh Cream
- Lards+Compounds
- Vinegar
- Breakfast Cereals
- Defined Milk+Cream Prd(B)

N.B. The category field used in the data is `rst_4_market`.

### 4. Apply specific gravity adjustment

The above food categories have volume measured in litres, and have have varying densities such that the water density approximation (1L ~ 1kg) is insufficiently accurate. Currently this adjustment is applied at the `rst_4_market_sector` category level and uses guidlines found here [To add]

### 5. Remove data entry errors

There exist some outlying products in the data that are now deemed to be data entry errors. Those removed include those that appear to have an energy density of greater than 900 kcal/100g (impossible) and those with a volume of 0. This is best practice although the impact on the result is negligible: 0.0075% of the total products are removed.

## Detail: Quantifying the impact of changes

The natural metrics for quantfying the impact of the changes above is through their effects on:

- The mean calorie reduction per person per day in the simulation outcome
- The npm reduction for each store predicted in the simulated outcome

The table below shows the impact of each of these updated steps (cumulatively):

[to add]

## Detail: Understanding the relationship of the input data with the results of the DSotN report

The published mean kcal consumption of food bought for purchase in-home is x. The flowchart below shows how this number becomes the y input to the retailer targets model.

## Detail: Additional robustness considerations

...
