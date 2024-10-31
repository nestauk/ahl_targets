"""
This file creates a new data file for the targets model (update October 2024). For information on the updates applied and reasoning refer to the README.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.getters import simulated_outcomes as get_sim_data
import yaml


from ahl_targets.utils import diets
from ahl_targets.getters import get_data_v2 as g2
import logging

# Variables for use in the script

adult_pop = 51718632
no_days = 365

# Product categories (rst_4_market) added into the new file that weren't in the old file
to_keep = [
    "Cooking Oils",
    "Total Ice Cream",
    "Fresh Cream",
    "Lards+Compounds",
    "Vinegar",
    "Breakfast Cereals",
    "Defined Milk+Cream Prd(B)",
]

# Specific gravity mapping that wasn't in original specific gravity file

# Taken from here: https://www.foodstandards.gov.au/business/labelling/nutrition-panel-calculator/specific-gravities

specific_gravity_mapping = {
    "Cooking Oils": 0.92,
    "Total Ice Cream": 0.55,
    "Fresh Cream": 1.01,
    "Lards+Compounds": 1,  # Assumption
    "Vinegar": 1.01,
    "Breakfast Cereals": 1,  # Assumption
    "Defined Milk+Cream Prd(B)": 1.07,  # Taken for evaporated milk
}


# Function for adjusting specific gravity of litres
def custom_function(row):
    if row["rst_4_market"] in to_keep:
        return row["volume_up"] * specific_gravity_mapping.get(row["rst_4_market"], 1)
    else:
        return row["volume_up"]


if __name__ == "__main__":

    logging.info("This script takes about 10 minutes to run")
    logging.info("Building new data file")

    # Read in the  new model file (created here: ahl_diets_evidence/pipeline/number_calories_gb_retailer_checks.py)
    df = g2.new_model_data()

    # Merge on NPM score
    npm = get_data.full_npm()

    df_npm = df.merge(
        npm[["purchase_id", "period", "npm_score", "kcal_per_100g"]],
        on=["purchase_id", "period"],
        how="left",
    )

    # Ensure unique products IDs
    df_npm["unique_id"] = df_npm["purchase_id"].astype(str) + df_npm["period"].astype(
        str
    )

    # Rename variables the equivalent in the old model
    df_npm = df_npm.rename(
        columns={
            "panel_id": "Panel Id",
            "gross_up_weight": "Gross Up Weight",
            "volume": "volume_up",
            "store_level_3": "store_cat",
            "energy_kcal": "Energy KCal",
            "quantity": "Quantity",
            "spend": "Spend",
        }
    )

    logging.info("Adjusting kcal and volume values to reflect adult intake")

    # Get adult intake

    demog_df = g2.get_demographics_data()
    adult_intake = diets.adult_intake(demog_df)

    # Get total prop intake for each household
    adult_intake = adult_intake.groupby("Panel Id")["prop_intake"].sum().reset_index()

    # Merge adult intake to store_data
    df_npm = df_npm.merge(
        adult_intake[["Panel Id", "prop_intake"]], on="Panel Id", how="left"
    )

    ##Adjust kcal and volume values to reflect adult intake

    # Set original names to "old"
    df_npm = df_npm.rename(
        columns={"Energy KCal": "old_kcal", "volume_up": "old_volume_up"}
    )

    # Adjust kcal and volume by adult prop intake
    df_npm["Energy KCal"] = df_npm["old_kcal"] * df_npm["prop_intake"]
    df_npm["volume_up"] = df_npm["old_volume_up"] * df_npm["prop_intake"]
    df_npm["weighted_kcal"] = df_npm["Energy KCal"] * df_npm["Gross Up Weight"]

    # Drop products that weren't in the old file (and aren't intentially added back in)

    added_surprise = g2.get_products_to_drop()

    df_npm = df_npm[~df_npm["unique_id"].isin(added_surprise["unique_id"].astype(str))]

    logging.info(
        f"Total kcal in new file: {df_npm['weighted_kcal'].sum() / adult_pop / no_days}"
    )

    # Calculate weighted NPM dataframe

    logging.info("Calculating weighted NPM dataframe (model input)")

    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(df_npm)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    print(store_weight_npm["total_kcal"].sum() / adult_pop / no_days)

    logging.info("Adjusting volume for specific gravity of litres")

    # Adjust volume for specific gravity of litres
    df_npm["adjusted_volume"] = df_npm.apply(lambda row: custom_function(row), axis=1)

    # Rename volume adjusted column to volume_up
    df_adj = df_npm.drop(columns=["volume_up"]).rename(
        columns={"adjusted_volume": "volume_up"}
    )

    store_weight_npm_adj = su.weighted_npm(df_adj)
    store_weight_npm_adj["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    save_prompt = input(
        "Would you like to save and overwrite the existing model on S3? (y/n)"
    )

    if save_prompt == "y":

        logging.info("Saving the new data files to S3")
        # Save non volume adjusted to S3

        upload_obj(
            df_npm,
            BUCKET_NAME,
            "in_home/processed/targets/oct_24_update/df_npm.parquet",
            kwargs_writing={"index": False},
        )

        upload_obj(
            store_weight_npm,
            BUCKET_NAME,
            "in_home/processed/targets/oct_24_update/store_weight.parquet",
            kwargs_writing={"index": False},
        )

        # Save volume adjusted to S3

        upload_obj(
            df_adj,
            BUCKET_NAME,
            "in_home/processed/targets/oct_24_update/df_npm_adj.parquet",
            kwargs_writing={"index": False},
        )

        upload_obj(
            store_weight_npm_adj,
            BUCKET_NAME,
            "in_home/processed/targets/oct_24_update/store_weight_adj.parquet",
            kwargs_writing={"index": False},
        )
