"""
This file calculates the sales-weighted average (SWA) NPM score by store for the original data and the 2024 update.

Flag these files are so big that they can cause the kernel to crash. Easy work around is to run for old/new one at a time.
"""

# Imports
import logging
from ahl_targets.getters import get_data_v2 as g2
from ahl_targets.getters import simulated_outcomes as so
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
import numpy as np
import gc

if __name__ == "__main__":
    # Get data
    logging.info("Reading in data")
    new_output_detailed = g2.model_results(detailed=True)

    new_swa_by_store = (
        (
            new_output_detailed.groupby(["store_cat", "iter"])
            .apply(lambda x: np.average(x["npm_score"], weights=x["new_total_kg"]))
            .reset_index()
            .rename(columns={0: "post_swa_npm"})
        )
        .groupby("store_cat")
        .mean()
        .reset_index()
        .drop(columns="iter")
        .merge(
            new_output_detailed.groupby(["store_cat", "iter"])
            .apply(lambda x: np.average(x["npm_score"], weights=x["total_kg"]))
            .reset_index()
            .rename(columns={0: "pre_swa_npm"})
        )
        .groupby("store_cat")
        .mean()
        .reset_index()
        .drop(columns="iter")
    )

    # Upload to S3
    logging.info("Uploading new swa by store to S3")
    upload_obj(
        new_swa_by_store,
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/swa_by_store_new.csv",
        kwargs_writing={"index": False},
    )

    # Clear memory
    logging.info("Clearing memory")
    del new_output_detailed
    gc.collect()

    # Read in data (loaded in seperately to help with memory issues)
    logging.info("Reading in old data")
    old_output_detailed = so.npm_agg(detailed=True)

    old_swa_by_store = (
        (
            old_output_detailed.groupby(["store_cat", "iter"])
            .apply(lambda x: np.average(x["npm_score"], weights=x["new_total_kg"]))
            .reset_index()
            .rename(columns={0: "post_swa_npm"})
        )
        .groupby("store_cat")
        .mean()
        .reset_index()
        .drop(columns="iter")
        .merge(
            old_output_detailed.groupby(["store_cat", "iter"])
            .apply(lambda x: np.average(x["npm_score"], weights=x["total_kg"]))
            .reset_index()
            .rename(columns={0: "pre_swa_npm"})
        )
        .groupby("store_cat")
        .mean()
        .reset_index()
        .drop(columns="iter")
    )

    # Upload to S3
    logging.info("Uploading old swa by store to S3")
    upload_obj(
        old_swa_by_store,
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/swa_by_store_old.csv",
        kwargs_writing={"index": False},
    )
