from ahl_targets.getters import get_data
from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets.pipeline import product_transformation as product
from nesta_ds_utils.loading_saving.S3 import upload_obj
import logging
from ahl_targets import BUCKET_NAME


if __name__ == "__main__":
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    pur_rec_vol = get_data.purchase_records_volume()
    prod_table = get_data.product_metadata()
    npm = get_data.get_npm()

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
    # subset to stores that are part of this analysis
    stores_sub = store_levels[store_levels["store_cat"].isin(keep_stores)]

    logging.info("merge purchase record with product info")
    dat1 = pur_rec_vol.merge(
        prod_table, left_on="Product Code", right_on="product_code"
    )

    logging.info("retain food only")
    dat1 = product.is_food(dat1).query("is_food == 1")

    logging.info("filter to kg only")
    # filter to kg only
    dat2 = dat1.query("reported_volume == 'Kilos'")

    logging.info("merge with store data and npm")

    # Merge with npm and store names
    store_data = dat2.merge(
        stores_sub, left_on="Store Code", right_on="store_id"
    ).merge(
        npm[["purchase_id", "period", "npm_score", "kcal_per_100g"]],
        left_on=["PurchaseId", "Period"],
        right_on=["purchase_id", "period"],
    )

    store_data.rename(columns={"kcal_per_100g": "ed"}, inplace=True)

    # remove implausible values
    store_data = store_data[store_data["ed"] < 900].copy()

    upload_obj(
        store_data,
        BUCKET_NAME,
        "in_home/processed/targets/model_data.csv",
        kwargs_writing={"index": False},
    )
