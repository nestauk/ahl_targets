"""This file does two things:
- Merges the new values back to the original file to check that for each unique purchase (defined by purchase_id and period) the key values are the same
(ie merging + transfer from diets) has worked as expected

On investigation, I found that there were unexpected products in the new file that weren't in the old file.
These were products that were in categories that were intentionally added back in.
The majority of these products had missing NPM scores, so were likely excluded in the original analysis for this reason.
~100 had, and it's hard to tell why they weren't included. However, this is a negligible number accounting for <1kcal pp per day, so I've removed them and saved an updated file.
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

    # Read in data

    orig_data = get_data.model_data()
    df_npm = pd.read_parquet(f"{PROJECT_DIR}/outputs/data/df_npm_new.parquet")

    # Ensure unique products IDs
    df_npm["unique_id"] = df_npm["purchase_id"].astype(str) + df_npm["period"].astype(
        str
    )
    orig_data["unique_id"] = orig_data["purchase_id"].astype(str) + orig_data[
        "period"
    ].astype(str)

    # Update the new data file with an indicator of whether it was in the original analysis
    df_npm["is_in_old"] = df_npm["unique_id"].isin(orig_data["unique_id"])

    # Get the same products and check ED and NPM score differences at product level - all matches

    store_data_comp = orig_data.merge(
        df_npm[
            ["purchase_id", "period", "npm_score", "kcal_per_100g", "old_volume_up"]
        ],
        on=["purchase_id", "period"],
        how="left",
        suffixes=("", "_new"),
    )

    store_data_comp["npm_diff"] = (
        store_data_comp["npm_score"] - store_data_comp["npm_score_new"]
    )
    store_data_comp["volume_diff"] = (
        store_data_comp["volume_up"] - store_data_comp["old_volume_up"]
    )
    store_data_comp["ed_diff"] = (
        store_data_comp["ed"] - store_data_comp["kcal_per_100g"]
    )

    logging.info(
        f"Number of records with different NPM: {store_data_comp[store_data_comp['npm_diff'] != 0].shape[0]}"
    )
    logging.info(
        f"Number of records with different volume: {store_data_comp[store_data_comp['volume_diff'] != 0].shape[0]}"
    )
    logging.info(
        f"Number of records with different ed: {store_data_comp[store_data_comp['ed_diff'] != 0].shape[0]}"
    )

    # Remove products that weren't in the categories added back in and weren't in the old model

    # List of categories intentionally added to original targets file
    to_keep = [
        "Cooking Oils",
        "Total Ice Cream",
        "Fresh Cream",
        "Lards+Compounds",
        "Vinegar",
        "Breakfast Cereals",
        "Defined Milk+Cream Prd(B)",
    ]

    # Filter for products that weren't in the old model, or in the categories to add
    added_new = df_npm[~df_npm["is_in_old"]]
    added_surprise = added_new[~added_new["rst_4_market"].isin(to_keep)]

    # The majority of these have missing NPM scores (likely why they were excluded in the original)

    logging.info(f"Total surprise additions: {added_surprise.shape[0]}")
    logging.info(
        f"Number dropped due to missing NPM scores: {added_surprise['npm_score'].isna().sum()}"
    )

    # Overall kcal per person per day is minimal

    logging.info(
        f"Total kcal per person per day of surprise additions: {added_surprise['energy_kcal_weighted'].sum()/adult_pop/no_days}"
    )

    # Therefore, just remove them and check the baseline effect
    logging.info("Removing surprise additions and saving updated file")

    df_npm = df_npm[~df_npm["unique_id"].isin(added_surprise["unique_id"])]
    added_new = added_new[~added_new["unique_id"].isin(added_surprise["unique_id"])]

    df_npm.to_parquet(f"{PROJECT_DIR}/outputs/data/df_npm_new_no_surprise.parquet")

    logging.info(f"New baseline is: {df_npm['weighted_kcal'].sum()/adult_pop/no_days}")
