# Scatter plot of products with avg NPM as x-axis and energy density as y-axis, coloured by HFSS.

# Import modules
from ahl_targets.pipeline import (
    product_transformation as product,
    energy_density as energy,
    hfss,
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

# Remove limit on rows for altair
alt.data_transformers.enable("default", max_rows=None)


def npm_ed_product_scatter(
    product_metrics_df: pd.DataFrame,
    data_subset: str,
    size_col: str,
    size_list: list,
    scale_list: list,
):
    """Plots scatter of npm vs energy density for product df.
    Args:
        product_metrics_df (pd.Dataframe): dataframe of products with npm, hfss and energy density info
        data_subset (str): Name of product category subset if using for plot title (eg 'chocolate confectionery)
        size_col (str): Name of column to use as size of circle
        size_list (list): Width and height of plot
        scale_list (list): Min and max values for circles
    Returns:
        Altair scatter plot
    """
    product_metrics_df["hfss"] = (
        product_metrics_df["in_scope"] * product_metrics_df[size_col]
    )
    scatter_plot_df = (
        product_metrics_df[["npm_score", "ed_deciles", size_col, "hfss"]]
        .copy()
        .groupby(["npm_score", "ed_deciles"])
        .sum()
        .reset_index()
    )
    scatter_plot_df["perc_hfss"] = (
        scatter_plot_df["hfss"] / scatter_plot_df[size_col] * 100
    )
    scatter_plot_df["Percent " + size_col] = (
        scatter_plot_df[size_col] / scatter_plot_df[size_col].sum()
    ) * 100

    scatter_plot_df = scatter_plot_df[scatter_plot_df["Percent " + size_col] > 0].copy()

    fig = (
        alt.Chart(scatter_plot_df)
        .mark_circle()
        .encode(
            x=alt.X("npm_score", title="NPM score"),
            y=alt.Y("ed_deciles", title="Energy density deciles"),
            size=alt.Size(
                "Percent " + size_col,
                title="Percent " + size_col,
                scale=alt.Scale(range=scale_list),
            ),
            color=alt.Color(
                "perc_hfss",
                scale=alt.Scale(scheme="purplegreen", domainMid=0.00001),
                title="Percent HFSS",
            ),
        )
        .properties(width=size_list[0], height=size_list[1])
    )
    return configure_plots(
        fig,
        data_subset + "Avg Energy Density vs Avg NPM score - Percent " + size_col,
        " ",
        16,
        14,
        14,
    )


if __name__ == "__main__":
    logging.info("Running metrics_compare.py, takes about 10 mins to run.")

    logging.info("Loading data...")
    # Get data
    nut_recs = get_data.nutrition()
    pur_recs = get_data.purchase_records_updated()
    pur_rec_vol = get_data.purchase_records_volume()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    prod_table = get_data.product_metadata()
    prod_meas = get_data.product_measurement()
    gravity = get_data.get_gravity()
    fvn = get_data.get_fvn()

    logging.info("Creating Energy density table...")
    # Create product level energy density table
    df_prod_ed = energy.prod_energy_100(
        "rst_4_market",
        pur_recs,
        nut_recs,
        prod_table,
        prod_meas,
        gravity,
    )
    df_prod_ed["ed_deciles"] = energy.decile(
        df_prod_ed["kcal_100g"],
    )
    logging.info("Creating NPM table...")
    # Create NPM table
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

    logging.info("Merging tables...")
    # Merge tables and create subset
    pur_prod_info = npm.merge(
        df_prod_ed,
        how="inner",
        left_on="product_code",
        right_on="product_code",
    ).drop("product_code", axis=1)
    pur_prod_info = product.in_scope(
        pur_prod_info.copy(),
    )

    logging.info("Adding kcal and product count...")
    # Clean purchases and merge with nutrition data
    pur_nut = tables.nutrition_merge(
        nut_recs,
        pur_recs[pur_recs["Reported Volume"].notna()].copy(),
        ["Energy KCal"],
    )[["Product Code", "Energy KCal"]].copy()
    pur_prod_info = pur_prod_info.copy().merge(
        pur_nut.groupby(["Product Code"]).sum().reset_index(),
        how="left",
        on="Product Code",
    )
    pur_prod_info["products"] = 1

    # Confectionery subset
    conf_purchases = pur_prod_info[
        pur_prod_info.rst_4_market_sector == "Take Home Confectionery"
    ].copy()

    logging.info("Creating and saving plots...")
    # Running function to plot data
    webdr = google_chrome_driver_setup()

    # All products - prod count
    save_altair(
        npm_ed_product_scatter(pur_prod_info, "", "products", [700, 500], [10, 600]),
        "ed_npm_all_prods",
        driver=webdr,
    )

    # Example category - prod count
    save_altair(
        npm_ed_product_scatter(
            conf_purchases,
            "Chocoloate confectionery: ",
            "products",
            [400, 300],
            [40, 600],
        ),
        "ed_npm_chocolate_conf",
        driver=webdr,
    )

    # All products - perc kcal
    save_altair(
        npm_ed_product_scatter(pur_prod_info, "", "Energy KCal", [700, 500], [10, 600]),
        "ed_npm_all_prods_kcal",
        driver=webdr,
    )

    # Example category - perc kcal
    save_altair(
        npm_ed_product_scatter(
            conf_purchases,
            "Chocoloate confectionery: ",
            "Energy KCal",
            [400, 300],
            [40, 600],
        ),
        "ed_npm_chocolate_conf_kcal",
        driver=webdr,
    )
