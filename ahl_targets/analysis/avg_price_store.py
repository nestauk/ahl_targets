from ahl_targets.getters import get_data
from ahl_targets.pipeline import stores_transformation as stores
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR
from ahl_targets import BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import upload_obj


if __name__ == "__main__":
    prod_table = get_data.product_metadata()
    pur_rec_vol = get_data.purchase_records_volume()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()

    store_levels = stores.custom_taxonomy(
        stores.taxonomy(
            store_coding,
            store_lines,
        )
    )

    # Subset by store
    store_subset = stores.store_subset(store_levels)

    # Merge with purchase records
    pur_rec_store = pur_rec_vol.merge(
        store_subset[["store_id", "store_cat"]],
        left_on="Store Code",
        right_on="store_id",
        how="left",
    ).dropna(subset=["store_cat"])

    # Calculate the weighted average product per store_cat
    weighted_prods_sold = (
        pur_rec_store.groupby(["store_cat", "Product Code"])
        .apply(
            lambda x: (x["Spend"] * x["Gross Up Weight"]).sum()
            / x["Gross Up Weight"].sum()
        )
        .reset_index()
    )
    weighted_prods_sold.columns = ["store_cat", "product", "Weighted_Avg_Spend"]
    # Group by store_cat
    weighted_prods_sold.groupby(["store_cat"])[
        "Weighted_Avg_Spend"
    ].mean().reset_index()

    # Calculate the weighted average price per store_cat
    weighted_purchases_store = (
        pur_rec_store.groupby("store_cat")
        .apply(
            lambda x: (x["Spend"] * x["Gross Up Weight"]).sum()
            / x["Gross Up Weight"].sum()
        )
        .reset_index()
    )
    weighted_purchases_store.columns = ["store_cat", "Weighted_Avg_Price"]

    # Save to S3
    upload_obj(
        weighted_prods_sold,
        BUCKET_NAME,
        "in_home/data_outputs/weighted_prods_sold.csv",
        kwargs_writing={"index": False, "encoding": "utf-8"},
    )
    upload_obj(
        weighted_purchases_store,
        BUCKET_NAME,
        "in_home/data_outputs/weighted_purchases_store.csv",
        kwargs_writing={"index": False, "encoding": "utf-8"},
    )
