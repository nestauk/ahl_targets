# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: ahl_targets
#     language: python
#     name: python3
# ---

# +
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

# -

to_keep = [
    "Cooking Oils",
    "Total Ice Cream",
    "Fresh Cream",
    "Lards+Compounds",
    "Vinegar",
    "Breakfast Cereals",
    "Defined Milk+Cream Prd(B)",
]

prod_table = get_data.product_metadata()

# Read in data
agg_df = g2.get_agg_data()
agg_df_adj = g2.get_agg_data_vol_adjusted()


def get_swas(store_weight_npm, prod_metadata=prod_table):

    # Calculate SWA NPM
    store_weight_npm["swa_npm"] = (
        store_weight_npm["kg_w"] * store_weight_npm["npm_score"]
    )

    # Calculate SWA NPM by market
    total_swa_npm = store_weight_npm["swa_npm"].sum()

    # Calculate SWA NPM by store
    store_weight_npm["kg_w_store_cat"] = store_weight_npm[
        "kg_w"
    ] / store_weight_npm.groupby("store_cat")["kg_w"].transform("sum")
    store_weight_npm["swa_npm_store_cat"] = (
        store_weight_npm["kg_w_store_cat"] * store_weight_npm["npm_score"]
    )
    store_swa_npm = (
        store_weight_npm["swa_npm_store_cat"]
        .groupby(store_weight_npm["store_cat"])
        .sum()
    )

    # Merge back markets
    store_weight_npm = store_weight_npm.merge(
        prod_metadata[["product_code", "rst_4_market", "rst_4_extended"]],
        on=["product_code"],
        how="left",
    )

    # Calculate SWA NPM by rst_4_market
    store_weight_npm["kg_w_rst_4_market"] = store_weight_npm[
        "kg_w"
    ] / store_weight_npm.groupby("rst_4_market")["kg_w"].transform("sum")
    store_weight_npm["swa_npm_rst_4_market"] = (
        store_weight_npm["kg_w_rst_4_market"] * store_weight_npm["npm_score"]
    )
    rst_4_market_swa_npm = (
        store_weight_npm["swa_npm_rst_4_market"]
        .groupby(store_weight_npm["rst_4_market"])
        .sum()
    )

    return total_swa_npm, store_swa_npm, rst_4_market_swa_npm, store_weight_npm


# +
# Import original data
# I did this to check that the SWA function works as expected (produces the same results as the original data). Have commented out the import to reduce the memory usage in the notebook
# Original results saved here: https://docs.google.com/spreadsheets/d/1ED3rxbJzZNi6lgRLUvwsL0NWz3FLSko4G6oJ2ohbSM0/edit?gid=0#gid=0)

# orig_data = get_data.model_data()

# store_weight_npm_orig = su.weighted_npm(orig_data)
# store_weight_npm_orig["prod_weight_g"] = store_weight_npm_orig.pipe(su.prod_weight_g)


# Check the SWA NPM for the original file
# total_swa_npm_orig, swa_by_store_orig, swa_by_market, agg_df_orig = get_swas(store_weight_npm_orig)

# +
# Compare SWA for volume-adjusted and non-volume adjusted data

total_swa, swa_by_store, swa_by_market, swa_all = get_swas(agg_df)
total_swa_adj, swa_by_store_adj, swa_by_market_adj, swa_all_adj = get_swas(agg_df_adj)

# +
# Plot SWA NPM values for volume-adjusted added categories

swa_by_market_adj = swa_by_market_adj.reset_index()

display(swa_by_market_adj[swa_by_market_adj["rst_4_market"].isin(to_keep)])

print(f"Total SWA NPM for volume-adjusted data: {total_swa_adj}")

# +
# Get count of HFSS products with and within added categories
agg_df_adj["hfss"] = agg_df_adj["npm_score"] > 4

agg_df_adj["hfss"].value_counts()

agg_df_adj[~agg_df_adj["rst_4_market"].isin(to_keep)]["hfss"].value_counts()
