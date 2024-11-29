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
