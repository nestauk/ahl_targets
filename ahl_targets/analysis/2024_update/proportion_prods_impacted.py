"""
This file calculates the proportion of products impacted by interventions (reformulation/sales shifts) by each model.

Flag these files are so big that they can cause the kernel to crash. Easy work around is to run for old/new one at a time.
"""

# Imports
import logging
from ahl_targets.getters import get_data_v2 as g2
from ahl_targets.getters import get_data
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import simulated_outcomes as so
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
import pandas as pd
import gc

if __name__ == "__main__":
    # Get data
    logging.info("Reading in new model data")
    new_output_detailed = g2.model_results(detailed=True)

    ## Proportion reformulated

    # Proportion of products reformulated by calories (i.e., proportion of calories reformulated)
    avg_kcal_reform = new_output_detailed.groupby("iter").apply(
        lambda x: (x["indicator_reform"] * x["total_kcal"]).sum()
        / (x["total_kcal"].sum())
    )
    logging.info(
        "Proportion of products reformulated by kcal (new model): {}".format(
            avg_kcal_reform.mean()
        )
    )

    # ii) By volume

    # Proportion of products reformulated by weight
    avg_weight_reform = new_output_detailed.groupby("iter").apply(
        lambda x: (x["indicator_reform"] * x["total_kg"]).sum() / (x["total_kg"].sum())
    )
    logging.info(
        "Proportion of products reformulated by weight (new model): {}".format(
            avg_weight_reform.mean()
        )
    )

    logging.info("Removing new model data and clearing memory")
    del new_output_detailed
    gc.collect()

    # Read in old model data
    logging.info("Reading in old model data")
    old_output_detailed = so.npm_agg(detailed=True)

    # Proportion of products reformulated by calories (i.e., proportion of calories reformulated)
    avg_kcal_reform = old_output_detailed.groupby("iter").apply(
        lambda x: (x["indicator_reform"] * x["total_kcal"]).sum()
        / (x["total_kcal"].sum())
    )
    logging.info(
        "Proportion of products reformulated by kcal (old model): {}".format(
            avg_kcal_reform.mean()
        )
    )

    # ii) By volume

    # Proportion of products reformulated by weight
    avg_weight_reform = old_output_detailed.groupby("iter").apply(
        lambda x: (x["indicator_reform"] * x["total_kg"]).sum() / (x["total_kg"].sum())
    )
    logging.info(
        "Proportion of products reformulated by weight (old model): {}".format(
            avg_weight_reform.mean()
        )
    )

    logging.info("Removing old model data and clearing memory")
    del old_output_detailed
    gc.collect()

    ## Proportion experiencing sales shifts (high and low)

    # Note: All products experience sales shifts with the current paramaters (product_share_sales_values == 1). So this is just the HFSS/non-HFSS split in the input data.

    # Read in input model data
    # Old
    store_data = get_data.model_data()
    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(store_data)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    old_model_data = store_weight_npm.copy()

    # New
    new_model_data = g2.get_agg_data_2024()

    # Calculate proportion HFSS
    for data, model in zip([old_model_data, new_model_data], ["old", "new"]):
        high_npm = data[data["npm_score"] >= 4].copy()
        logging.info(
            f"Proportion of products with high NPM (negatively sales-shifted) by weight ({model} model): {high_npm['total_kg'].sum() / data['total_kg'].sum()}"
        )
        logging.info(
            f"Proportion of products with high NPM (negatively sales-shifted) by kcal ({model} model): {high_npm['total_kcal'].sum() / data['total_kcal'].sum()}"
        )

    ## Saving results

    # Create dataframe of outputs (reformulation and sales shifts)
    output_table = pd.DataFrame(
        {
            "model": ["new", "old"],
            "reformulation_kcal": [avg_kcal_reform.mean(), avg_kcal_reform.mean()],
            "reformulation_weight": [
                avg_weight_reform.mean(),
                avg_weight_reform.mean(),
            ],
            "sales_shifts_kcal": [
                high_npm["total_kcal"].sum() / data["total_kcal"].sum(),
                high_npm["total_kcal"].sum() / data["total_kcal"].sum(),
            ],
            "sales_shifts_weight": [
                high_npm["total_kg"].sum() / data["total_kg"].sum(),
                high_npm["total_kg"].sum() / data["total_kg"].sum(),
            ],
        }
    )

    # Upload to S3
    logging.info("Uploading outputs to S3")
    upload_obj(
        output_table,
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/proportion_products_impacted.csv",
        kwargs_writing={"index": False},
    )

# As expected, the proportion of products reformulated and sales-shifted is higher in the new model, as the input data contains more HFSS products.
