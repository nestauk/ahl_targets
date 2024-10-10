"""
This script aims to prepare the data by applying the same exclusions as in diets:
1) Cutting out Jan-Mar data
2) Leaving drinks in
3) Leaving products with (ed<900) in - Flag - if this is making the change we probably shouldn't switch to doing this!
4) Leaving all stores in
"""

from ahl_targets.getters import get_data
from ahl_targets.utils import stores_transformation as stores
from ahl_targets.utils import product_transformation as product
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets.utils.io import load_with_encoding
import logging
from ahl_targets import BUCKET_NAME
import pandas as pd
from ahl_targets import PROJECT_DIR


# Getters for files that have local paths
def get_store_itemisation_line() -> pd.DataFrame:
    """Reads the dataset of store lines."""
    return pd.read_csv(
        load_with_encoding(
            BUCKET_NAME, "in_home/latest_data/store_itemisation_lines.csv"
        ),
        encoding="ISO-8859-1",
    )


def get_store_itemisation_coding() -> pd.DataFrame:
    """Reads the dataset of store codes."""
    return pd.read_csv(
        load_with_encoding(
            BUCKET_NAME, "in_home/latest_data/store_itemisation_coding.csv"
        ),
        encoding="ISO-8859-1",
    )


if __name__ == "__main__":
    store_coding = get_store_itemisation_coding()
    store_lines = get_store_itemisation_line()
    pur_rec_vol = get_data.purchase_records_volume()
    prod_table = get_data.product_metadata()
    npm = get_data.full_npm()
    nut = (
        get_data.nutrition()
    )  # Flag: Have downloaded this file from : https://eu-west-2.console.aws.amazon.com/s3/object/ahl-private-data?region=eu-west-2&bucketType=general&prefix=in_home/latest_data/nutrition_data.csv

    logging.info("This script takes about 20 minutes to run")

    # define stores to keep
    keep_stores = [
        "Total Tesco",
        "Total Sainsbury's",
        "Total Asda",
        "Total Morrisons",
        "Aldi",
        "Lidl",
        "Total Waitrose",
        "The Co-Operative",
        "Total Marks & Spencer",
        "Total Iceland",
        "Ocado Internet",
    ]
    # create custom taxonomy
    store_levels = stores.custom_taxonomy(
        stores.taxonomy(
            store_coding,
            store_lines,
        )
    )

    ### 4) Leaving all stores in ###
    # subset to stores that are part of this analysis
    # stores_sub = store_levels[store_levels["store_cat"].isin(keep_stores)]

    # stores that are not part of the analysis
    stores_not_sub = store_levels[~store_levels["store_cat"].isin(keep_stores)]

    logging.info("merge purchase record with product info")
    dat1 = pur_rec_vol.merge(
        prod_table, left_on="Product Code", right_on="product_code"
    )

    # 2) Leaving drinks in
    # logging.info("retain food only")
    # dat1 = product.is_food(dat1).query("is_food == 1")

    logging.info("filter to kg only")
    # filter to kg only
    dat2_original = dat1.query("reported_volume == 'Kilos'")

    # For info - how many products are filtered?
    logging.info(
        f"% of sales volume filtered out: {1 - (dat2_original.Quantity * dat2_original.Volume * dat2_original['Gross Up Weight']).sum() / (dat1.Quantity * dat1.Volume * dat1['Gross Up Weight']).sum()}"
    )
    # 57% filtered!! Is this right!? This will also filter out drinks as it filters out L

    # Let's include L and say 1L = 1kg
    dat2 = dat1.query(
        "(reported_volume == 'Kilos') | (reported_volume == 'Litres')"
    ).copy()

    logging.info(
        f"% of sales volume filtered out: {1 - (dat2.Quantity * dat2.Volume * dat2['Gross Up Weight']).sum() / (dat1.Quantity * dat1.Volume * dat1['Gross Up Weight']).sum()}"
    )
    # 5% filtered

    logging.info("merge with store data and npm")

    # Merge with npm and store names (model data)
    store_data = dat2.merge(
        store_levels, left_on="Store Code", right_on="store_id"
    ).merge(
        npm[["purchase_id", "period", "npm_score", "kcal_per_100g"]],
        left_on=["PurchaseId", "Period"],
        right_on=["purchase_id", "period"],
    )

    store_data.rename(columns={"kcal_per_100g": "ed"}, inplace=True)

    # remove implausible values (not removing anymore but keeping into see how many are stripped)
    store_data_ed = store_data[store_data["ed"] < 900].copy()

    logging.info(
        f'% of sales volume filtered by ed < 900: {1 - (store_data_ed.Quantity * store_data_ed.Volume * store_data_ed["Gross Up Weight"]).sum() / (store_data.Quantity * store_data.Volume * store_data["Gross Up Weight"]).sum()}'
    )

    # Cutting out Jan-Mar data (1)
    # Filter data from April 2021 onwards
    store_data = store_data[
        pd.to_datetime(store_data["Purchase Date"], format="%d/%m/%Y") >= "01/04/2021"
    ]

    # Local save store data at this point for crashes
    store_data.to_csv(
        PROJECT_DIR / "outputs/model_data_unfiltered_pre.csv", index=False
    )

    out = store_data.merge(
        nut[["Purchase Number", "Purchase Period", "Energy KCal"]],
        left_on=["PurchaseId", "Period"],
        right_on=["Purchase Number", "Purchase Period"],
    )

    # # Merge with npm and store names (NOT model data)
    # store_data_not_sub = dat2.merge(
    #     stores_not_sub, left_on="Store Code", right_on="store_id"
    # ).merge(
    #     npm[["purchase_id", "period", "npm_score", "kcal_per_100g"]],
    #     left_on=["PurchaseId", "Period"],
    #     right_on=["purchase_id", "period"],
    # )

    # store_data_not_sub.rename(columns={"kcal_per_100g": "ed"}, inplace=True)

    # # remove implausible values
    # store_data_not_sub = store_data_not_sub[store_data_not_sub["ed"] < 900].copy()

    # out_not_sub = store_data_not_sub.merge(
    #     nut[["Purchase Number", "Purchase Period", "Energy KCal"]],
    #     left_on=["PurchaseId", "Period"],
    #     right_on=["Purchase Number", "Purchase Period"],
    # )

    # upload_obj(
    #     out,
    #     BUCKET_NAME,
    #     "in_home/processed/targets/model_data.parquet",
    #     kwargs_writing={"compression": "zstd", "engine": "pyarrow"},
    # )

    # upload_obj(
    #     out_not_sub,
    #     BUCKET_NAME,
    #     "in_home/processed/targets/excluded_data.parquet",
    #     kwargs_writing={"compression": "zstd", "engine": "pyarrow"},
    # )

    # Local save
    out.to_csv(PROJECT_DIR / "outputs/model_data_unfiltered.csv", index=False)
