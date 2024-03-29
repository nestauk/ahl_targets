# Import modules
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

# Remove limit on rows for altair
alt.data_transformers.enable("default", max_rows=None)


def category_avg_df(
    avg_df: pd.DataFrame, category_df: pd.DataFrame, category_name: str
) -> pd.DataFrame:
    """Combines the category and total dfs containing grouped kcal results.
    Args:
        avg_df (pd.Dataframe): dataframe of average grouped kcal
        category_df (pd.Dataframe): dataframe of category grouped kcal
        category_name (str): Category name to use
    Returns:
        pd.DataFrame: combined category and total kcal info
    """
    category_df["type"] = category_name
    avg_df["type"] = "All products"
    return pd.concat([category_df, avg_df], axis=0)


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
    df_prod_ed["ed_deciles"] = (
        energy.decile(
            df_prod_ed["kcal_100g"],
        )
        + 1
    )
    return df_prod_ed


def perc_kcal(pur_df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Creates gross kcal and percent of kcal df grouped by specified category.
    Args:
        pur_df (pd.Dataframe): dataframe with kcal, weights and category info per purchase
        category_col (str): Name of category column to use
    Returns:
        pd.DataFrame: gross and percent kcal grouped by category
    """
    gross_kcal = calorie_sales.kcal_sales(
        pur_df,
        category_col,
    )
    gross_kcal["percent"] = gross_kcal["gross_kcal"] / gross_kcal["gross_kcal"].sum()
    return gross_kcal


def kcal_percent_bar_plot(
    pur_store_info: pd.DataFrame,
    category_col: str,
    category_name: str,
    colour: str,
    num_cats: int,
    plot_size: list,
):
    """Plots percent of kcal per specified category as bar chart.
    Args:
        pur_store_info (pd.Dataframe): dataframe with kcal, weights and category info per purchase
        category_col (str): Name of category column to use
        category_name (str): Name of category to use as chart titles
        colour (str): Hex colour code to use
        num_cats (int): Number of category values to show in plot
        plot_size (list): Width and height of plot as list
    """
    percent_kcal_df = perc_kcal(
        pur_store_info,
        category_col,
    )
    base = (
        alt.Chart(
            percent_kcal_df.sort_values(by="percent", ascending=False).head(num_cats)
        )
        .encode(
            x=alt.X("percent", title="Percentage", axis=alt.Axis(format="%")),
            y=alt.Y(category_col, sort="-x", title=category_name),
            color=alt.value(colour),
            text=alt.Text("percent", format=".1%"),
        )
        .properties(width=plot_size[0], height=plot_size[1])
    )
    fig = base.mark_bar() + base.mark_text(align="left", dx=2)
    return save_altair(
        configure_plots(
            fig,
            "Percent of kcal " + category_name,
            " ",
            16,
            14,
            14,
        ),
        category_name + "_bar_kcal",
        driver=webdr,
    )


def kcal_percent_bar_plot_stacked(
    pur_store_info: pd.DataFrame,
    category_col: str,
    sub_category_col: str,
    category_name: str,
    num_cats: int,
    plot_size: list,
):
    """Plots percent of kcal per specified category as bar chart - stacked by type.
    Args:
        pur_store_info (pd.Dataframe): dataframe with kcal, weights and category info per purchase
        category_col (str): Name of category column to use
        sub_category_col (str): Name of sub category column to use
        category_name (str): Name of category to use as chart titles
        colour (str): Hex colour code to use
        num_cats (int): Number of category values to show in plot
        plot_size (list): Width and height of plot as list
    """
    percent_kcal_df = perc_kcal(
        pur_store_info,
        [category_col, sub_category_col],
    )
    bars = (
        alt.Chart(
            percent_kcal_df.sort_values(by="percent", ascending=False).head(num_cats)
        )
        .mark_bar()
        .encode(
            x=alt.X("sum(percent)", title="Percentage", axis=alt.Axis(format="%")),
            y=alt.Y(category_col, sort="-x", title=category_name),
            color=sub_category_col,
        )
        .properties(width=plot_size[0], height=plot_size[1])
    )

    text = (
        alt.Chart(
            percent_kcal_df.sort_values(by="percent", ascending=False).head(num_cats)
        )
        .mark_text(align="right", dx=30)
        .encode(
            y=alt.Y(category_col, sort="-x", title=category_name),
            x=alt.X("sum(percent)", title="Percentage", axis=alt.Axis(format="%")),
            text=alt.Text("sum(percent)", format=".1%"),
        )
    )

    fig = bars + text

    return save_altair(
        configure_plots(
            fig,
            "Percent of kcal " + category_name,
            " ",
            16,
            14,
            14,
        ),
        category_name + "_bar_kcal",
        driver=webdr,
    )


def pie_percent_kcal(
    pur_store_info: pd.DataFrame, category_col: str, category_name: str
):
    """Plots percent of kcal per specified category as pie chart.
    Args:
        pur_store_info (pd.Dataframe): dataframe with kcal, weights and category info per purchase
        category_col (str): Name of category column to use
        category_name (str): Name of category to use as chart titles
    """
    percent_kcal_df = perc_kcal(
        pur_store_info,
        category_col,
    )
    percent_kcal_df["percent"] = (percent_kcal_df["percent"] * 100).round(0)
    base = (
        alt.Chart(percent_kcal_df)
        .mark_arc()
        .encode(
            theta=alt.Theta("percent", stack=True),
            color=category_col + ":N",
        )
    )

    pie = base.mark_arc(innerRadius=20, stroke="#fff")
    text = base.mark_text(radiusOffset=10, radius=60, size=16, fill="white").encode(
        text="percent:Q"
    )

    fig = pie + text
    return save_altair(
        configure_plots(
            fig,
            "Percent of kcal " + category_name,
            " ",
            16,
            14,
            14,
        ),
        category_name + "_pie_kcal",
        driver=webdr,
    )


def plots_energy_category(
    ed_kcal: pd.DataFrame, cat_kcal_df: pd.DataFrame, category_name: str
):
    """Plots total kcal and percent of kcal across energy deciles and category in two plots.
    Args:
        ed_kcal (pd.Dataframe): dataframe with total energy density kcal info
        cat_kcal_df (pd.Dataframe): dataframe with kcal per category
        category_name (str): Name of category to use as chart titles
    """
    cat_ed_kcal = perc_kcal(
        cat_kcal_df,
        "ed_deciles",
    )
    # Create df for plotting
    ed_cat_df = category_avg_df(
        ed_kcal,
        cat_ed_kcal,
        category_name,
    )
    # Bar chart total kcal
    bar_kcal = (
        alt.Chart(cat_ed_kcal)
        .mark_bar()
        .encode(
            x=alt.X("ed_deciles:N", title="Energy density deciles"),
            y=alt.Y("gross_kcal:Q", title="Total kcal"),
            color=alt.value("#EB003B"),
        )
        .properties(width=300, height=200)
    )
    bar_kcal_plot = configure_plots(
        bar_kcal,
        "Total kcal across energy density deciles: " + category_name,
        " ",
        16,
        14,
        14,
    )

    bar_percent = (
        alt.Chart(ed_cat_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("ed_deciles:N", title="Energy density deciles"),
            y=alt.Y("percent:Q", title="Percent", axis=alt.Axis(format="%")),
            color="type:N",
        )
        .properties(width=300, height=300)
    )
    bar_percent_plot = configure_plots(
        bar_percent,
        "Percent kcal "
        + category_name
        + " compared to all products across energy density deciles",
        " ",
        16,
        14,
        14,
    )
    return save_altair(
        bar_kcal_plot,
        category_name + "_energy_density_bar_kcal",
        driver=webdr,
    ), save_altair(
        bar_percent_plot,
        category_name + "_energy_density_bar_perc",
        driver=webdr,
    )


def plot_inscope_bar(is_kcal: pd.DataFrame, df: pd.DataFrame, category_name: str):
    """Plots inscope / not inscope per category compared to average.
    Args:
        is_kcal (pd.Dataframe): dataframe with total inscope kcal info
        df (pd.Dataframe): dataframe with kcal per category
        category_name (str): Name of category to use as chart titles
    """
    cat_is_kcal = perc_kcal(
        df,
        "in_scope",
    )
    combined_df = category_avg_df(
        is_kcal,
        cat_is_kcal,
        category_name,
    )
    fig = (
        alt.Chart(combined_df)
        .mark_bar()
        .encode(
            x="type:O",
            y=alt.Y("percent:Q", axis=alt.Axis(format="%")),
            color="type:N",
            column="in_scope:N",
        )
        .properties(width=70, height=200)
    )
    return save_altair(
        configure_plots(
            fig,
            "Percent kcal " + category_name + " compared to all products in_scope",
            " ",
            16,
            14,
            14,
        ),
        category_name + "_inscope_bar",
        driver=webdr,
    )


def plot_npm_bar(category_df: pd.DataFrame, npm_kcal: pd.DataFrame, category_name: str):
    """Plots npm per category compared to average.
    Args:
        category_df (pd.Dataframe): dataframe with kcal per category
        npm_kcal (pd.Dataframe): dataframe with total npm kcal info
        category_name (str): Name of category to use as chart titles
    """
    category_df_food = category_df[category_df.is_food == 1].copy()
    cat_npm_kcal = perc_kcal(
        category_df_food,
        "npm_score",
    )
    combined_df = category_avg_df(
        npm_kcal,
        cat_npm_kcal,
        category_name,
    )

    fig = (
        alt.Chart(combined_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("npm_score:Q", title="NPM score"),
            y=alt.Y("percent:Q", title="Percent", axis=alt.Axis(format="%")),
            color="type:N",
        )
        .properties(width=500, height=300)
    )

    line = (
        alt.Chart(pd.DataFrame({"x": [4]}))
        .mark_rule()
        .encode(x="x", strokeDash=alt.value([5, 5]))
    )

    figure = fig + line

    return save_altair(
        configure_plots(
            figure,
            "Percent of kcal per NPM scores - " + category_name,
            " ",
            16,
            14,
            14,
        ),
        category_name + "_npm_bar",
        driver=webdr,
    )


def heatmap_category_kcal(
    pur_info: pd.DataFrame,
    category_col: str,
    metric_col: str,
    plot_size: list,
    file_name: str,
    cat_name: str,
    metric_name: str,
):
    """Plots heatmap of % kcal for category and metric.
    Args:
        pur_info (pd.Dataframe): dataframe
        category_col (str): Name of category column
        metric_col (str): Name of metric column
        plot_size (list): Width and height of the plot
        file_name (str): Name of file
        cat_name (str): Name of category (for axis title)
        metric_name (str) Name of metric (for axis title)
    """

    pur_store_perc = (
        pur_info.groupby([category_col, metric_col])["gross_kcal"].sum().reset_index()
    )
    # Fill missing NPM scores as zero
    categories = list(range(-10, 31))
    mux = pd.MultiIndex.from_product(
        [pur_store_perc["itemisation_level_3"].unique(), categories],
        names=("itemisation_level_3", "npm_score"),
    )
    pur_store_perc = (
        pur_store_perc.set_index(["itemisation_level_3", "npm_score"])
        .reindex(mux, fill_value=0)
        .swaplevel(0, 1)
        .reset_index()
    )
    # Create percent field
    pur_store_perc["Percent Kcal"] = (
        100
        * pur_store_perc["gross_kcal"]
        / pur_store_perc.groupby(category_col)["gross_kcal"].transform("sum")
    ).round(1)

    heat = (
        alt.Chart(pur_store_perc)
        .mark_rect()
        .encode(
            x=alt.X(metric_col + ":N", title=metric_name),
            y=alt.Y(category_col, title=cat_name),
            color=alt.Color("Percent Kcal:Q"),
        )
        .properties(width=plot_size[0], height=plot_size[1])
    )

    text = (
        alt.Chart(pur_store_perc)
        .mark_text(baseline="middle")
        .encode(
            x=alt.X(metric_col + ":N"),
            y=alt.Y(category_col),
            text="Percent Kcal:Q",
            color=alt.condition(
                datum["Percent Kcal"] < (pur_store_perc["Percent Kcal"].max()) / 2,
                alt.value("black"),
                alt.value("white"),
            ),
        )
    )

    fig = heat + text

    return save_altair(
        configure_plots(
            fig,
            "Percent of kcal - " + cat_name + " / " + metric_name,
            " ",
            16,
            14,
            14,
        ),
        file_name,
        driver=webdr,
    )


if __name__ == "__main__":
    logging.info("Running kcal_sales_distribution.py, takes about 10 mins to run.")

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

    logging.info("Transforming data ready for plots...")
    logging.info("- Creating energy density dataframe")
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
    # Recoding hard discounters as aldi and lidl
    store_levels["itemisation_level_3"] = np.where(
        store_levels["itemisation_level_4"] == "Aldi",
        "Aldi",
        store_levels["itemisation_level_3"],
    )
    store_levels["itemisation_level_3"] = np.where(
        store_levels["itemisation_level_4"] == "Lidl",
        "Lidl",
        store_levels["itemisation_level_3"],
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

    logging.info("- Creating subset dfs for plotting")
    # Create dfs for plotting
    ed_kcal = perc_kcal(
        pur_store_info.copy(),
        "ed_deciles",
    )
    breakfast_df = pur_store_info[
        pur_store_info.rst_4_market == "Breakfast Cereals"
    ].copy()
    co_op_df = pur_store_info[
        pur_store_info.itemisation_level_3 == "The Co-Operative"
    ].copy()
    walkers_df = pur_store_info[
        pur_store_info.manufacturer == "Walkers Snack Foods Ltd"
    ].copy()
    is_kcal = perc_kcal(
        pur_store_info.dropna(subset=["npm_score"]).copy(),
        "in_scope",
    )
    pur_store_info_food = pur_store_info[pur_store_info.is_food == 1].copy()
    npm_kcal = perc_kcal(
        pur_store_info_food.dropna(subset=["npm_score"]).copy(),
        "npm_score",
    )

    # Load web-driver
    webdr = google_chrome_driver_setup()

    pur_store_info["gross_kcal"] = (
        pur_store_info["Energy KCal"] * pur_store_info["Gross Up Weight"]
    )

    # Create and save plots

    # Food and drink categories
    kcal_percent_bar_plot_stacked(
        pur_store_info,
        "rst_4_market",
        "type",
        "category (top 30)",
        30,
        [450, 600],
    )

    # Brand
    kcal_percent_bar_plot_stacked(
        pur_store_info,
        "manufacturer",
        "type",
        "brand (top 30)",
        30,
        [450, 600],
    )
    # Store
    kcal_percent_bar_plot_stacked(
        pur_store_info,
        "itemisation_level_3",
        "type",
        "store",
        100,
        [350, 500],
    )
    # Store - tesco
    kcal_percent_bar_plot(
        pur_store_info[pur_store_info.itemisation_level_3 == "Total Tesco"].copy(),
        "itemisation_level_4",
        "tesco",
        "#FDB633",
        100,
        [200, 200],
    )
    # Instore / online pie
    pie_percent_kcal(
        pur_store_info,
        "online",
        "online-instore",
    )
    # Breakfast cereals
    breakfast_kcal_plot, breakfast_percent_plot = plots_energy_category(
        ed_kcal,
        breakfast_df,
        "Breakfast Cereals",
    )
    # Co-op
    coop_kcal_plot, coop_percent_plot = plots_energy_category(
        ed_kcal,
        co_op_df,
        "Co-op",
    )
    # Walkers
    walkers_kcal_plot, walkers_percent_plot = plots_energy_category(
        ed_kcal,
        walkers_df,
        "Walkers",
    )
    # Pie - type
    pie_percent_kcal(
        pur_store_info.dropna(subset=["npm_score"]).copy(),
        "type",
        "type",
    )
    # Pie - inscope
    pie_percent_kcal(
        pur_store_info.dropna(subset=["npm_score"]).copy(),
        "in_scope",
        "in scope",
    )
    # Co-op bar inscope
    plot_inscope_bar(
        is_kcal,
        co_op_df.dropna(subset=["npm_score"]).copy(),
        "Co-op",
    )
    # Breakfast bar inscope
    plot_inscope_bar(
        is_kcal,
        breakfast_df.dropna(subset=["npm_score"]).copy(),
        "Breakfast Cereals",
    )
    # Walkers bar inscope
    plot_inscope_bar(
        is_kcal,
        walkers_df.dropna(subset=["npm_score"]).copy(),
        "Walkers",
    )
    # Co-op npm bar
    plot_npm_bar(
        co_op_df.dropna(subset=["npm_score"]).copy(),
        npm_kcal,
        "Co-op",
    )
    # Walkers npm bar
    plot_npm_bar(
        walkers_df.dropna(subset=["npm_score"]).copy(),
        npm_kcal,
        "Walkers",
    )
    # Breakfast npm bar
    plot_npm_bar(
        breakfast_df.dropna(subset=["npm_score"]).copy(),
        npm_kcal,
        "Breakfast Cereals",
    )

    # Store heatmaps
    heatmap_category_kcal(
        pur_store_info,
        "itemisation_level_3",
        "ed_deciles",
        [300, 400],
        "ed_store_heat",
        "Store",
        "Energy density deciles",
    )
    heatmap_category_kcal(
        pur_store_info,
        "itemisation_level_3",
        "npm_score",
        [700, 400],
        "npm_store_heat",
        "Store",
        "NPM score",
    )
