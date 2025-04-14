"""
This file compares the outputs of the 2024 update with the original model and produces summary tables.

The comparison is done for the following metrics:
- Baseline avg calorie per person per day
- Baseline sales-weighted avg NPM score
- New avg calorie per person per day (post retailer targets simulation)
- New sales-weighted avg NPM score (post retailer targets simulation)
- Calorie reduction predicted by the model

We then investigate the differences.
"""

# Imports
import logging
import pandas as pd
import numpy as np
from ahl_targets.getters import get_data_v2 as g2
from ahl_targets.getters import get_data
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import simulated_outcomes as so
from ahl_targets import PROJECT_DIR, BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import download_obj
from boto3.s3.transfer import TransferConfig

if __name__ == "__main__":
    # Get data
    logging.info("Reading in data")
    orig_output = so.npm_agg()
    new_output = g2.model_results()

    # Calc output table
    logging.info("Calculating output table")
    output_table = pd.DataFrame(
        {
            "metric": [
                "kcal_pp_baseline",
                "swa_npm_baseline",
                "kcal_pp_post",
                "swa_npm_post",
                "predicted_calorie_reduction",
            ],
            "new_model": [
                new_output["kcal_pp_baseline"].mean().round(2),
                new_output["mean_npm_kg_baseline"].mean().round(2),
                new_output["kcal_pp_new"].mean().round(2),
                new_output["mean_npm_kg_new"].mean().round(2),
                (new_output["kcal_pp_baseline"] - new_output["kcal_pp_new"])
                .mean()
                .round(2),
            ],
            "old_model": [
                orig_output["kcal_pp_baseline"].mean().round(2),
                orig_output["mean_npm_kg_baseline"].mean().round(2),
                orig_output["kcal_pp_new"].mean().round(2),
                orig_output["mean_npm_kg_new"].mean().round(2),
                (orig_output["kcal_pp_baseline"] - orig_output["kcal_pp_new"])
                .mean()
                .round(2),
            ],
        }
    )

    # Log output table
    logging.info("Output table")
    logging.info(output_table)

    # Save output table
    output_table.to_csv(
        PROJECT_DIR / "outputs/model_output_comparison.csv",
        index=False,
    )

    #### Differences investigation ####
    # 1. Differences in kcal_pp_baseline and swa_npm_baseline are known and understood due to the update in input data.
    # 2. The differences in the *effect* of the model is more surprising. The kcal reduction in the new model is much larger in the new model (74 vs 51) whereas the magnitude of the swa_npm difference is relatively small (1.07 vs 1.01). This suggests that some of the regression coefficients in the new data are larger.
    #    To identify whether this larger difference is credible we will investigate:
    #    i) The differences in swa_npm by store
    #    ii) The differences in the regression coefficients by category
    #    iii) The difference in %s of products that are reformulated/subject to a sales shift in each model

    # i) Differences in swa_npm by store
    logging.info("Loading swa by store data")  # Calculated in swa_by_store.py
    # new_swa_by_store = g2.get_swa_by_store()
    # old_swa_by_store = g2.get_swa_by_store(old=True)

    # Temp: use local data
    new_swa_by_store = pd.read_csv(PROJECT_DIR / "outputs/new_swa_by_store.csv")
    old_swa_by_store = pd.read_csv(PROJECT_DIR / "outputs/old_swa_by_store.csv")

    # Display the magnitude of npm reduction pre and post-simulation
    new_swa_by_store["npm_reduction"] = (
        new_swa_by_store["post_swa_npm"] - new_swa_by_store["pre_swa_npm"]
    )
    old_swa_by_store["npm_reduction"] = (
        old_swa_by_store["post_swa_npm"] - old_swa_by_store["pre_swa_npm"]
    )

    # Display the baseline NPM differences between the new and old input data by store
    new_swa_by_store["baseline_npm_diff"] = (
        new_swa_by_store["pre_swa_npm"] - old_swa_by_store["pre_swa_npm"]
    )

    # Log the results
    logging.info("New model swa by store")
    logging.info(new_swa_by_store)
    logging.info("Old model swa by store")
    logging.info(old_swa_by_store)

    # Insights:
    #    - The reductions are all of a similar magnitude by store, although in general the new model leads to slightly larger reductions. Mean reduction: 0.695 vs 0.664. Median reduction: 0.701 vs 0.666.
    #    - The new model has an increased baseline NPM for all stores. The most notable are Iceland, Co-op and Waitrose with baseline increases of 0.4. We would expect this to be as a result of relatively higher sales of re-included products.

    ## Verifying the increase in baseline NPM: Iceland, Co-op and Waitrose
    # Get the old and new model data
    logging.info("Loading old and new model data")

    # Old
    store_data = get_data.model_data()
    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(store_data)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    old_model_data = store_weight_npm.copy()

    # New
    new_model_data = g2.get_agg_data_2024()

    # Filter to the stores of interest
    store_filter = ["Total Iceland", "The Co-Operative", "Total Waitrose"]
    old_model_data = old_model_data[
        old_model_data["store_cat"].isin(store_filter)
    ].reset_index(drop=True)
    new_model_data = new_model_data[
        new_model_data["store_cat"].isin(store_filter)
    ].reset_index(drop=True)

    # Examine the new products
    new_products = new_model_data[
        ~new_model_data.product_code.isin(old_model_data.product_code.unique())
    ].reset_index(drop=True)

    # Group by store and calculate swa of NPM for new products and compare to the swa across all products
    new_products_swa = (
        new_products.groupby("store_cat")
        .apply(lambda x: np.average(x["npm_score"], weights=x["total_kg"]))
        .reset_index()
        .rename(columns={0: "new_products_swa"})
    )
    all_products_swa = (
        new_model_data.groupby("store_cat")
        .apply(lambda x: np.average(x["npm_score"], weights=x["total_kg"]))
        .reset_index()
        .rename(columns={0: "all_products_swa"})
    )
    new_products_swa = new_products_swa.merge(
        all_products_swa, on="store_cat", how="left"
    )

    # Log the results
    logging.info("New products swa by store")
    logging.info(
        new_products_swa
    )  # SWA of new products is *much* higher. We expect these to be oils, icecreams etc.

    ## Further investigation:
    # What exactly are these products?

    # Load the silver product data to merge on
    ## Flag: Due to python version issues I can't seem to use ahl core data - accessing the file directly from the s3 bucket instead
    prod = download_obj(
        BUCKET_NAME,
        "kantar/silver/inhome_products_silver_full.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )

    # Merge on prod
    new_products = new_products.merge(
        prod[
            [
                "product_code",
                "product_long_description",
                "rst_4_market",
                "rst_4_market_sector",
            ]
        ],
        on="product_code",
        how="left",
    )

    # For each store, aggregate the % of total_kg in each market sector
    new_product_categories_share = (
        new_products.groupby(["store_cat", "rst_4_market"])["total_kg"]
        .sum()
        .groupby(level=0, group_keys=False)
        .apply(lambda x: x / x.sum())
        .reset_index(name="share")
    )

    # Log the output
    logging.info("New product categories share by store")
    logging.info(new_product_categories_share)

    # Insights:
    # - As expected - the swa of the new products is *much* higher. It is unsurprising they significantly increase the baseline NPM.
    # - Visually inspecting the products: The majority are oils/ice creams as expected.
    # - Looking at the categories, ice cream is particularly impactful here. 48% of the re-included volume in co-up is "Total Ice Cream", and 75% in Iceland!

    # Conc: No unexpected results on SWA NPM given the re-inclusion of high NPM products. Current hypothesis: this is driving the increased kcal reduction in the new model.

    ############
