from ahl_targets.pipeline.model_data import model_data
from ahl_targets.getters import get_data
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR


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
    stores = model_data(
        prod_table, pur_rec_vol, gravity, nut_recs, fvn, store_coding, store_lines
    )
    # market value share
    spend_share = (
        (stores["Gross Up Weight"] * stores["Spend"]).groupby(stores["store_cat"]).sum()
        / (stores["Gross Up Weight"] * stores["Spend"]).sum()
    ).reset_index(name="spend_share")
    volume_share = (
        (stores["Gross Up Weight"] * stores["volume_up"])
        .groupby(stores["store_cat"])
        .sum()
        / (stores["Gross Up Weight"] * stores["volume_up"]).sum()
    ).reset_index(name="volume_share")
    product_share = (
        (stores["Gross Up Weight"]).groupby(stores["store_cat"]).sum()
        / (stores["Gross Up Weight"]).sum()
    ).reset_index(name="product_share")
    kcal_share = (
        (stores["Gross Up Weight"] * stores["Energy KCal"])
        .groupby(stores["store_cat"])
        .sum()
        / (stores["Gross Up Weight"] * stores["Energy KCal"]).sum()
    ).reset_index(name="kcal_share")
    ed_share = (
        (stores["Gross Up Weight"] * stores["ed"]).groupby(stores["store_cat"]).sum()
        / (stores["Gross Up Weight"]).groupby(stores["store_cat"]).sum()
    ).reset_index(name="ed_average")
    npm_share = (
        (stores["Gross Up Weight"] * stores["npm_score"])
        .groupby(stores["store_cat"])
        .sum()
        / (stores["Gross Up Weight"]).groupby(stores["store_cat"]).sum()
    ).reset_index(name="npm_average")
    hfss_share = (
        (stores["Gross Up Weight"] * stores["in_scope"])
        .groupby(stores["store_cat"])
        .sum()
        / (stores["Gross Up Weight"]).groupby(stores["store_cat"]).sum()
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
    store_info.to_csv(PROJECT_DIR / "outputs/reports/store_info.csv")
