# %%
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
from statsmodels.stats.weightstats import DescrStatsW

# %%


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


def make_weights(df: pd.DataFrame):
    df = df.dropna().copy()
    df["kcal_w"] = df["Gross Up Weight"] * df["Energy KCal"]
    df["kg_w"] = df["total_sale"]
    df["prod_w"] = df["Gross Up Weight"]
    return df


def make_cross(df_weight: pd.DataFrame, val: str, weight: str):
    df_weight[val + "_" + weight] = df_weight[val] * df_weight[weight]
    return df_weight.copy()


def desc_stat(df_weight: pd.DataFrame, val: str, weight=str):
    df_weight = df_weight.pipe(make_cross, val, weight).copy()
    return (
        DescrStatsW(df_weight[val], weights=df_weight[weight]).mean,
        DescrStatsW(df_weight[val], weights=df_weight[weight]).std,
    )


def agg_meas(
    df_weight: pd.DataFrame, agg: pd.Series, val: pd.Series, weight: pd.Series
):
    df_weight = df_weight.pipe(make_cross, val, weight).copy()
    return (
        df_weight.groupby([agg])[val + "_" + weight].sum()
        / df_weight.groupby([agg])[weight].sum()
    ).reset_index(name="_" + val)


def agg(dfs: list, on: pd.Series):
    data_frames = dfs
    return ft.reduce(lambda left, right: pd.merge(left, right, on=[on]), dfs)


def agg_data(agg_var: pd.Series, df_weight: pd.DataFrame, weight: str):
    return agg(
        [
            agg_meas(df_weight, agg_var, "in_scope", weight),
            agg_meas(df_weight, agg_var, "npm_score", weight),
            agg_meas(df_weight, agg_var, "kcal_100g", weight),
            agg_meas(df_weight, agg_var, "Energy KCal", weight),
        ],
        agg_var,
    ).copy()


def hist_agg(
    df_weight: pd.DataFrame, agg: str, var: str, weight: str, min: float, max: float
):
    fig = (
        alt.Chart(agg_meas(df_weight, agg, var, weight))
        .mark_bar()
        .encode(
            alt.X(
                "_" + var + ":Q",
                bin=True,
                scale=alt.Scale(domain=[min, max]),
                axis=alt.Axis(tickMinStep=1),
            ),
            alt.Y("count()"),
        )
    )

    return save_altair(
        configure_plots(
            fig,
            var + " by " + agg + "weight_" + weight,
            " ",
            16,
            14,
            14,
        ),
        var + " by " + agg + "_hist_" + weight,
        driver=webdr,
    )


# %%


def scatter_agg(df_weight: pd.DataFrame, agg: str, var1: str, var2: str, weight: str):
    fig = (
        alt.Chart(agg_data(agg, df_weight, weight))
        .mark_circle(size=60)
        .encode(x=var1, y=var2)
    )

    return save_altair(
        configure_plots(
            fig,
            var1 + " by " + var2 + "weight_" + weight,
            " ",
            16,
            14,
            14,
        ),
        var1 + " by " + var2 + "_scatter_" + weight,
        driver=webdr,
    )


# %%

# if __name__ == "__main__":
nut_recs = get_data.nutrition()
pur_recs = get_data.purchase_records_updated()
pur_rec_vol = get_data.purchase_records_volume()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()
prod_table = get_data.product_metadata()
prod_meas = get_data.product_measurement()
gravity = get_data.get_gravity()
fvn = get_data.get_fvn()

# %%

webdr = google_chrome_driver_setup()
# %%

df_prod_ed = energy_df(
    pur_recs,
    nut_recs,
    prod_table,
    prod_meas,
    gravity,
)
# %%

store_levels = stores.taxonomy(
    store_coding,
    store_lines,
)
# %%

# Clean purchases and merge with nutrition data
pur_nut = tables.nutrition_merge(
    nut_recs,
    pur_recs[pur_recs["Reported Volume"].notna()].copy(),
    ["Energy KCal"],
)
# %%

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
# %%

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
# %%

df_weight = pur_store_info.pipe(make_weights).copy()

# %%


hist_agg(df_weight, "itemisation_level_3", "in_scope", "kcal_w", 0, 0.8)

# %%

hist_agg(df_weight, "itemisation_level_3", "npm_score", "kcal_w", -5, 20)
hist_agg(df_weight, "itemisation_level_3", "Energy KCal", "kcal_w", 100, 90000)
hist_agg(df_weight, "itemisation_level_3", "kcal_100g", "kcal_w", 0, 500)

hist_agg(df_weight, "itemisation_level_3", "in_scope", "kg_w", 0, 0.8)
hist_agg(df_weight, "itemisation_level_3", "npm_score", "kg_w", -5, 20)
hist_agg(df_weight, "itemisation_level_3", "Energy KCal", "kg_w", 100, 90000)
hist_agg(df_weight, "itemisation_level_3", "kcal_100g", "kg_w", 0, 500)

hist_agg(df_weight, "itemisation_level_3", "in_scope", "prod_w", 0, 0.8)
hist_agg(df_weight, "itemisation_level_3", "npm_score", "prod_w", -5, 20)
hist_agg(df_weight, "itemisation_level_3", "Energy KCal", "prod_w", 100, 90000)
hist_agg(df_weight, "itemisation_level_3", "kcal_100g", "prod_w", 0, 500)

# %%

scatter_agg(pur_store_info, "itemisation_level_3", "_in_scope", "_npm_score")
scatter_agg(pur_store_info, "itemisation_level_3", "_in_scope", "_Energy KCal")
scatter_agg(pur_store_info, "itemisation_level_3", "_in_scope", "_kcal_100g")
scatter_agg(pur_store_info, "itemisation_level_3", "_npm_score", "_Energy KCal")
scatter_agg(pur_store_info, "itemisation_level_3", "_npm_score", "_kcal_100g")
scatter_agg(pur_store_info, "itemisation_level_3", "_kcal_100g", "_Energy KCal")

# %%
df_weight = pur_store_info.pipe(make_weights).copy()

# %%

pd.DataFrame(
    list(desc_stat(df_weight, "in_scope", "kcal_w")),
    index=["mean", "std"],
    columns=["in_scope"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "npm_score", "kcal_w")),
    index=["mean", "std"],
    columns=["npm_score"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "Energy KCal", "kcal_w")),
    index=["mean", "std"],
    columns=["Energy KCal"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "kcal_100g", "kcal_w")),
    index=["mean", "std"],
    columns=["kcal_100g"],
)

# %%
pd.DataFrame(
    list(desc_stat(df_weight, "in_scope", "kg_w")),
    index=["mean", "std"],
    columns=["in_scope"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "npm_score", "kg_w")),
    index=["mean", "std"],
    columns=["npm_score"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "Energy KCal", "kg_w")),
    index=["mean", "std"],
    columns=["Energy KCal"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "kcal_100g", "kg_w")),
    index=["mean", "std"],
    columns=["kcal_100g"],
)
# %%
desc_stat(df_weight, "in_scope", "prod_w")
desc_stat(df_weight, "npm_score", "prod_w")
desc_stat(df_weight, "Energy KCal", "prod_w")
desc_stat(df_weight, "kcal_100g", "prod_w")

# %%

pd.DataFrame(
    list(desc_stat(df_weight, "in_scope", "prod_w")),
    index=["mean", "std"],
    columns=["in_scope"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "npm_score", "prod_w")),
    index=["mean", "std"],
    columns=["npm_score"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "Energy KCal", "prod_w")),
    index=["mean", "std"],
    columns=["Energy KCal"],
)
# %%

pd.DataFrame(
    list(desc_stat(df_weight, "kcal_100g", "prod_w")),
    index=["mean", "std"],
    columns=["kcal_100g"],
)

# %%
df_weight["high_d"] = np.where(df_weight["ed_deciles"] > 8, 1, 0)
# %%
pd.crosstab(df_weight["high_d"], df_weight["in_scope"])
# %%
kilos = df_weight.groupby(["high_d", "in_scope"])["kg_w"].sum().reset_index()
# %%
kilos["share"] = kilos["kg_w"] / kilos["kg_w"].sum()
# %%
kilos
# %%
kcal = df_weight.groupby(["high_d", "in_scope"])["kcal_w"].sum().reset_index()
kcal["share"] = kcal["kcal_w"] / kcal["kcal_w"].sum()


# %%
kcal
# %%
prod = df_weight.groupby(["high_d", "in_scope"])["prod_w"].sum().reset_index()
prod["share"] = prod["prod_w"] / prod["prod_w"].sum()
# %%
prod
# %%
prod_count = df_weight.groupby(["high_d", "in_scope"]).size().reset_index(name="count")
prod_count["share"] = prod_count["count"] / prod_count["count"].sum()
# %%
prod_count
# %%


# %%
