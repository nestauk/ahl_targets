from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets.pipeline import product_transformation as product
from ahl_targets.pipeline import calorie_weighted_sales as calorie_sales
from ahl_targets.pipeline import energy_density as energy
from ahl_targets.pipeline import hfss
from ahl_targets.utils import create_tables as tables
from ahl_targets.getters import get_data
from ahl_targets.utils.plotting import configure_plots
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
from altair.expr import datum
import pandas as pd
import numpy as np
import altair as alt
import logging
import functools as ft


def energy_df(
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_table: pd.DataFrame,
    prod_meas: pd.DataFrame,
    gravity: pd.DataFrame,
) -> pd.DataFrame:
    """Creates energy density info dataframe per product.
    Args:
        pur_recs (pd.Dataframe): dataframe of purchase records
        nut_recs (pd.Dataframe): dataframe of nutrition records
        prod_table (pd.Dataframe): dataframe of product info
        prod_meas (pd.Dataframe): dataframe of product measurements
    Returns:
        pd.DataFrame: energy density info
    """
    # Create product level energy density table
    df_prod_ed = energy.prod_energy_100(
        "rst_4_market",
        pur_recs,
        nut_recs,
        prod_table,
        prod_meas,
        gravity,
    )
    df_prod_ed["energy_density_cat"] = energy.score(df_prod_ed["kcal_100g"])
    df_prod_ed["ed_deciles"] = energy.decile(
        df_prod_ed["kcal_100g"],
    )
    return df_prod_ed


def agg_meas(df: pd.DataFrame, agg: pd.Series, val: pd.Series):
    df["_" + val] = df[val] * df["total_sale"]
    return (
        df.groupby([agg])["_" + val].sum()
        / pur_store_info.groupby([agg])["total_sale"].sum()
    ).reset_index(name="_" + val)


def agg(dfs: list, on: pd.Series):
    data_frames = dfs
    return ft.reduce(lambda left, right: pd.merge(left, right, on=[on]), dfs)


def agg_data(agg_var: pd.Series, df: pd.DataFrame):
    return agg(
        [
            agg_meas(df, agg_var, "in_scope"),
            agg_meas(df, agg_var, "npm_score"),
            agg_meas(df, agg_var, "kcal_100g"),
            agg_meas(df, agg_var, "Energy KCal"),
        ],
        agg_var,
    ).copy()


def hist_agg(df: pd.DataFrame, agg: str, var: str, min: float, max: float):
    fig = (
        alt.Chart(agg_meas(df, agg, var))
        .mark_bar()
        .encode(alt.X("_" + var + ":Q", bin=True), alt.Y("count()", scale=alt.Scale(domain=[min, max])))
    )

    return save_altair(
        configure_plots(
            fig,
            var + " by " + agg,
            " ",
            16,
            14,
            14,
        ),
        var + " by " + agg + "_hist",
        driver=webdr,
    )


def scatter_agg(df: pd.DataFrame, agg: str, var1: str, var2: str):
    fig = alt.Chart(agg_data(agg, df)).mark_circle(size=60).encode(x=var1, y=var2)

    return save_altair(
        configure_plots(
            fig,
            var1 + " by " + var2,
            " ",
            16,
            14,
            14,
        ),
        var1 + " by " + var2 + "_scatter",
        driver=webdr,
    )


if __name__ == "__main__":
    nut_recs = get_data.nutrition()
    pur_recs = get_data.purchase_records_updated()
    pur_rec_vol = get_data.purchase_records_volume()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    prod_table = get_data.product_metadata()
    prod_meas = get_data.product_measurement()
    gravity = get_data.get_gravity()
    fvn = get_data.get_fvn()

    webdr = google_chrome_driver_setup()

    df_prod_ed = energy_df(
        pur_recs,
        nut_recs,
        prod_table,
        prod_meas,
        gravity,
    )

    store_levels = stores.taxonomy(
        store_coding,
        store_lines,
    )

    # Clean purchases and merge with nutrition data
    pur_nut = tables.nutrition_merge(
        nut_recs,
        pur_recs[pur_recs["Reported Volume"].notna()].copy(),
        ["Energy KCal"],
    )

    logging.info("- Creating NPM dataframe")
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
    logging.info("- Merging dataframes")
    # Merge product, store and household info

    pur_store_info = (
        pur_nut[["Store Code", "Product Code", "Gross Up Weight", "Energy KCal"]]
        .merge(
            store_levels[["store_id", "itemisation_level_3", "itemisation_level_4"]],
            how="left",
            left_on="Store Code",
            right_on="store_id",
        )
        .merge(
            df_prod_ed[
                [
                    "Product Code",
                    "rst_4_market",
                    "rst_4_market_sector",
                    "energy_density_cat",
                    "kcal_100g",
                    "ed_deciles",
                    "manufacturer",
                    "total_sale",
                ]
            ],
            how="left",
            left_on="Product Code",
            right_on="Product Code",
        )
        .merge(
            npm,
            how="left",
            left_on="Product Code",
            right_on="product_code",
        )
        .drop(["store_id", "Store Code", "product_code"], axis=1)
    )
    pur_store_info = stores.online(
        pur_store_info.copy(),
    )
    pur_store_info = product.in_scope(
        pur_store_info.copy(),
    )

    hist_agg(pur_store_info, "itemisation_level_3", "in_scope", 0, 0.8)
    hist_agg(pur_store_info, "itemisation_level_3", "npm_score", -5, 20)
    hist_agg(pur_store_info, "itemisation_level_3", "Energy KCal",100, 90000)
    hist_agg(pur_store_info, "itemisation_level_3", "kcal_100g", 0, 500)

    scatter_agg(pur_store_info, "itemisation_level_3", "_in_scope", "_npm_score")
    scatter_agg(pur_store_info, "itemisation_level_3", "_in_scope", "_Energy KCal")
    scatter_agg(pur_store_info, "itemisation_level_3", "_in_scope", "_kcal_100g")
    scatter_agg(pur_store_info, "itemisation_level_3", "_npm_score", "_Energy KCal")
    scatter_agg(pur_store_info, "itemisation_level_3", "_npm_score", "_kcal_100g")
    scatter_agg(pur_store_info, "itemisation_level_3", "_kcal_100g", "_Energy KCal")
