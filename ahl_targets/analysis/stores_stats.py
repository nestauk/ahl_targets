from ahl_targets.pipeline import model_data
from ahl_targets.getters import get_data
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR
import logging

prod_table = get_data.product_metadata()
pur_rec_vol = get_data.purchase_records_volume()
nut_recs = get_data.nutrition()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()
gravity = get_data.get_gravity()
fvn = get_data.get_fvn()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()


if __name__ == "__main__":
    store_data = model_data.make_data(
        prod_table, pur_rec_vol, gravity, nut_recs, fvn, store_coding, store_lines
    )

    # market value share
    spend_share = (
        (store_data["Gross Up Weight"] * store_data["Spend"])
        .groupby(store_data["store_cat"])
        .sum()
        / (store_data["Gross Up Weight"] * store_data["Spend"]).sum()
    ).reset_index(name="spend_share")
    volume_share = (
        (store_data["Gross Up Weight"] * store_data["volume_up"])
        .groupby(store_data["store_cat"])
        .sum()
        / (store_data["Gross Up Weight"] * store_data["volume_up"]).sum()
    ).reset_index(name="volume_share")
    product_share = (
        (store_data["Gross Up Weight"]).groupby(store_data["store_cat"]).sum()
        / (store_data["Gross Up Weight"]).sum()
    ).reset_index(name="product_share")
    kcal_share = (
        (store_data["Gross Up Weight"] * store_data["Energy KCal"])
        .groupby(store_data["store_cat"])
        .sum()
        / (store_data["Gross Up Weight"] * store_data["Energy KCal"]).sum()
    ).reset_index(name="kcal_share")
    ed_share = (
        (store_data["Gross Up Weight"] * store_data["ed"])
        .groupby(store_data["store_cat"])
        .sum()
        / (store_data["Gross Up Weight"]).groupby(store_data["store_cat"]).sum()
    ).reset_index(name="ed_average")
    npm_share = (
        (store_data["Gross Up Weight"] * store_data["npm_score"])
        .groupby(store_data["store_cat"])
        .sum()
        / (store_data["Gross Up Weight"]).groupby(store_data["store_cat"]).sum()
    ).reset_index(name="npm_average")
    hfss_share = (
        (store_data["Gross Up Weight"] * store_data["in_scope"])
        .groupby(store_data["store_cat"])
        .sum()
        / (store_data["Gross Up Weight"]).groupby(store_data["store_cat"]).sum()
    ).reset_index(name="hfss_average")

    datasets = [
        spend_share,
        volume_share,
        product_share,
        kcal_share,
        ed_share,
        npm_share,
        hfss_share,
    ]

    store_info = reduce(
        lambda left, right: pd.merge(left, right, on="store_cat", how="inner"), datasets
    )
    logging.info("Saving store_info.csv in outputs/reports")
    store_info.to_csv(PROJECT_DIR / "outputs/reports/store_info.csv")
