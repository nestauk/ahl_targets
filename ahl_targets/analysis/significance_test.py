# Import modules
from ahl_targets.pipeline import (
    product_transformation as product,
    energy_density as energy,
    hfss,
    stores_transformation as stores,
)
from ahl_targets.utils import create_tables as tables
from ahl_targets.getters import get_data
from ahl_targets.utils.plotting import configure_plots
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
import pandas as pd
import logging
import statsmodels.formula.api as smf
from patsy.contrasts import Sum
import numpy as np


def select_dat(
    dat: pd.DataFrame,
    npm: int,
):
    return dat[dat["npm_score"] == npm]


def make_dat(
    dat: pd.DataFrame,
    npm: int,
):
    sub_dat = select_dat(dat, npm)
    mean = sub_dat.groupby(["itemisation_level_3"])["share"].mean().reset_index()
    mean["mean"] = sub_dat["share"].mean()
    mean["std"] = sub_dat["share"].std()
    mean["upper_bound"] = mean["mean"] + mean["std"]
    mean["lower_bound"] = mean["mean"] - mean["std"]
    mean["over"] = np.where(mean["share"] >= mean["upper_bound"], 1, 0)
    mean["below"] = np.where(mean["share"] <= mean["lower_bound"], 1, 0)
    mean["sig"] = mean["over"] + mean["below"]
    return mean


def ref_line(
    dat: pd.DataFrame,
    npm: int,
):
    return pd.DataFrame([[select_dat(dat, npm)["share"].mean()]], columns=["share"])


def make_coef_plot(dat: pd.DataFrame, npm: int):
    points = (
        alt.Chart(make_dat(dat, npm), title="NPM =" + str(npm))
        .mark_circle(size=60)
        .encode(
            alt.Y(
                "itemisation_level_3:N",
                sort=alt.EncodingSortField(field="share", order="descending"),
            ),
            x="share",
            color="sig:N",
        )
    )

    rules = alt.Chart(ref_line(dat, npm)).mark_rule().encode(x="share")

    fig = points + rules
    return save_altair(
        configure_plots(
            fig,
            "NPM = " + str(npm),
            " ",
            16,
            14,
            14,
        ),
        "npm_" + str(npm),
        driver=webdr,
    )

    # %%


if __name__ == "__main__":
    logging.info("Reading data")
    nut_recs = get_data.nutrition()
    pur_recs = get_data.purchase_records_updated()
    pur_rec_vol = get_data.purchase_records_volume()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    prod_table = get_data.product_metadata()
    prod_meas = get_data.product_measurement()
    gravity = get_data.get_gravity()
    fvn = get_data.get_fvn()

    # Load web-driver
    webdr = google_chrome_driver_setup()

    logging.info("Generating NPM score")
    npm = hfss.npm_score_unique(
        prod_table,
        pur_rec_vol,
        gravity,
        nut_recs,
        fvn,
        hfss.a_points_cols(),
        "fiber_score",
        "protein_score",
        "Score",
    )

    logging.info("Get store types")
    store_levels = stores.taxonomy(
        store_coding,
        store_lines,
    )

    logging.info("Merge with nutritional file")

    tbl_nut = tables.nutrition_merge(nut_recs, pur_recs, ["Energy KCal"]).merge(
        store_levels, left_on=["Store Code"], right_on=["store_id"]
    )
    logging.info("Select variables")

    tbl_nut = tbl_nut[
        [
            "Panel Id",
            "Product Code",
            "Store Code",
            "PurchaseId",
            "Period",
            "Gross Up Weight",
            "itemisation_level_1",
            "itemisation_level_2",
            "itemisation_level_3",
            "itemisation_level_4",
            "Energy KCal",
        ]
    ].copy()
    # calculate contribution to total kcal sold at retailer for each product

    tbl_nut["gross_kcal"] = tbl_nut["Gross Up Weight"] * tbl_nut["Energy KCal"]

    logging.info("Merge with NPM")

    dat = tbl_nut.merge(npm, left_on=["Product Code"], right_on=["product_code"])

    num = (
        dat.groupby(["itemisation_level_3", "npm_score"])["gross_kcal"]
        .sum()
        .reset_index(name="sum_kcal")
    )
    den = (
        tbl_nut.groupby(["itemisation_level_3"])["gross_kcal"]
        .sum()
        .reset_index(name="total")
    )

    df = num.merge(den, on=["itemisation_level_3"])

    df["share"] = df["sum_kcal"] / df["total"] * 100

    for i in range(-10, 20):
        make_coef_plot(df, i)
