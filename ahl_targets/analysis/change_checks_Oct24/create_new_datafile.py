"""
This file creates a new data file for the model, which includes the NPM score and adjusts the kcal and volume values to reflect adult intake from the diets work

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

if __name__ == "__main__":

    adult_pop = 51718632
    no_days = 365

    logging.info("This script takes about 5 minutes to run")
    logging.info("Building new data file")

    # Build the new model file
    df = g2.new_model_data()

    # Merge on NPM score
    npm = get_data.full_npm()

    df_npm = df.merge(
        npm[["purchase_id", "period", "npm_score", "kcal_per_100g"]],
        on=["purchase_id", "period"],
        how="left",
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

    logging.info("Calculating weighted NPM dataframe (model input)")

    store_data = get_data.model_data()
    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(df_npm)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    print(store_weight_npm["total_kcal"].sum() / adult_pop / no_days)

    logging.info("Saving  the new data file locally for checking")
    # Store locally for further checking
    df_npm.to_parquet(f"{PROJECT_DIR}/outputs/data/df_npm_new.parquet")
    store_weight_npm.to_parquet(
        f"{PROJECT_DIR}/outputs/data/store_weight_npm_new.parquet"
    )
