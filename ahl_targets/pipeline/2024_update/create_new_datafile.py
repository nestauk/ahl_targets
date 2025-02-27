"""
2024 Update: This script generates the new input data file for the retailer targets model. For detailed information on the updates applied and reasoning refer to the README: `ahl_targets/pipeline/2024_update/README.md`.
"""

from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.utils import diets
from ahl_targets.getters import get_data_v2 as g2
import logging

# Define variables
adult_pop = 51718632
no_days = 365

if __name__ == "__main__":
    logging.info("Building model data. This script takes about 10 minutes to run")
    # Read in the new model file (details on how this is created can be found in the diets repository: https://github.com/nestauk/ahl_diets_evidence/blob/45_targets_data/ahl_diets_evidence/pipeline/number_calories_gb_retailer_checks.py)
    df = g2.new_model_data()

    # Merge on NPM scores
    npm = get_data.full_npm()

    df_npm = df.merge(
        npm[["purchase_id", "period", "npm_score"]],
        on=["purchase_id", "period"],
        how="left",
    )

    # Create unique products IDs
    df_npm["unique_id"] = df_npm["purchase_id"].astype(str) + df_npm["period"].astype(
        str
    )

    # Quick fix: Rename variables the same names as in the old model. This allows us to reuse the same functions.
    df_npm = df_npm.rename(
        columns={
            "panel_id": "Panel Id",
            "gross_up_weight": "Gross Up Weight",
            "adjusted_volume": "volume_up",
            "store_level_3": "store_cat",
            "energy_kcal": "Energy KCal",
            "quantity": "Quantity",
            "spend": "Spend",
        }
    )

    ## Apply adult intake conversion factors
    logging.info("Adjusting kcal and volume values to reflect adult intake")

    # Load demographic data and calculate adult intake conversion factors
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

    #### Update: This section drops a few products that didn't appear in the original file. Going to keep them in for now, not sure why they were dropped.

    # Drop products that weren't in the old file (and aren't intentially added back in)

    added_surprise = g2.get_products_to_drop()

    df_npm = df_npm[~df_npm["unique_id"].isin(added_surprise["unique_id"].astype(str))]

    logging.info(
        f"Kcal pp per day baseline: {df_npm['weighted_kcal'].sum() / adult_pop / no_days}"
    )

    ## Calculate weighted NPM dataframe
    logging.info("Calculating weighted NPM dataframe (model input)")

    # Load product data
    prod_table = get_data.product_metadata()

    # Calculate weighted npm for each store
    store_weight_npm = su.weighted_npm(df_npm)

    # Calculate product weights in grams
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    logging.info(
        "NPM SWA baseline: {}".format(
            (store_weight_npm["npm_score"] * store_weight_npm["kg_w"]).sum()
        )
    )

    save_prompt = input(
        "Would you like to save and overwrite the existing model on S3? (y/n)"
    )

    if save_prompt == "y":
        logging.info("Saving the new data files to S3")

        upload_obj(
            df_npm,
            BUCKET_NAME,
            "in_home/processed/targets/oct_24_update/df_npm.parquet",
            kwargs_writing={"compression": "zstd", "engine": "pyarrow"},
        )

        upload_obj(
            store_weight_npm,
            BUCKET_NAME,
            "in_home/processed/targets/oct_24_update/store_weight.parquet",
            kwargs_writing={"compression": "zstd", "engine": "pyarrow"},
        )
